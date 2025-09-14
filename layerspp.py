# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file
"""Layers for defining NCSN++.
"""
import layers
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

conv1x1 = layers.ddpm_conv1x1
conv3x3 = layers.ddpm_conv3x3
default_init = layers.default_init
naive_upsample = layers.naive_upsample
naive_downsample = layers.naive_downsample

class FourierFeatures(nn.Module):
    def __init__(self, first=5.0, last=6.0, step=1.0):
        super().__init__()
        self.freqs_exponent = torch.arange(first, last + 1e-8, step)

    @property
    def num_features(self):
        return len(self.freqs_exponent) * 2

    def forward(self, x):
        assert len(x.shape) >= 2

        # Compute (2pi * 2^n) for n in freqs.
        freqs_exponent = self.freqs_exponent.to(dtype=x.dtype, device=x.device)  # (F, )
        freqs = 2.0**freqs_exponent * 2 * torch.pi  # (F, )
        freqs = freqs.view(-1, *([1] * (x.dim() - 1)))  # (F, 1, 1, ...)

        # Compute (2pi * 2^n * x) for n in freqs.
        features = freqs * x.unsqueeze(1)  # (B, F, X1, X2, ...)
        features = features.flatten(1, 2)  # (B, F * C, X1, X2, ...)

        # Output features are cos and sin of above. Shape (B, 2 * F * C, H, W).
        return torch.cat([features.sin(), features.cos()], dim=1)
    
class GaussianFourierProjection(nn.Module):
  """Gaussian Fourier embeddings for noise levels."""

  def __init__(self, embedding_size=256, scale=1.0):
    super().__init__()
    self.W = nn.Parameter(torch.randn(embedding_size) * scale, requires_grad=False)

  def forward(self, x):
    x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
    return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)



class SelfAttentionBlock(nn.Module):
    """
    Compute self attention over channels, with skip connection.
    """
    def __init__(self, channels, init_scale=1e-2, dimensions=2):
        """

        :param channels:
        :param init_scale: A number between 0 and 1 which scales the variance of initialisation
            of the output layer. 1 is the usual Glorot initialisation. The default is 0.01 so that
            the output of the attention head is squashed in favor of the skip connection at the beginning of training.
        """
        super(SelfAttentionBlock, self).__init__()
        if dimensions == 1:
            conv = nn.Conv1d
        elif dimensions == 2:
            conv = nn.Conv2d
        elif dimensions == 3:
            conv = nn.Conv3d
        assert (init_scale <= 1) and (init_scale > 0)
        self.to_qkv = conv(in_channels=channels, out_channels=3*channels, kernel_size=1)
        self.to_out = conv(in_channels=channels, out_channels=channels, kernel_size=1)

        # initialisation
        with torch.no_grad():
            bound = init_scale / channels ** (1 / 2)
            self.to_out.weight.uniform_(-bound, bound)
            self.to_out.bias.zero_()
            bound = 1 / channels ** (1 / 2)
            self.to_qkv.weight.uniform_(-bound, bound)
            self.to_qkv.bias.zero_()

    def __call__(self, x):
        B, C, *D = x.shape
        q, k, v = torch.tensor_split(self.to_qkv(x), 3, dim=1)

        q = q.permute(0, *range(2, len(D)+2), 1).view(B, np.prod(D), C)
        k = k.view(B, C, np.prod(D))
        v = v.permute(0, *range(2, len(D)+2), 1).view(B, np.prod(D), C)

        w = torch.bmm(q, k) * (C**(-0.5))  # scaled channel attention matrix, QK^T / sqrt(d)
        w = torch.softmax(w, dim=-1)
        attention = torch.bmm(w, v)
        attention = attention.view(B, *D, C).permute(0, -1, *range(1, len(D)+1))
        return self.to_out(attention) + x


class ResnetBlockBigGANpp(nn.Module):
  def __init__(self, act, in_ch, out_ch=None, temb_dim=None, up=False, down=False,
               dropout=0.1, fir=False, fir_kernel=(1, 3, 3, 1),
               skip_rescale=True, init_scale=0.):
    super().__init__()

    out_ch = out_ch if out_ch else in_ch
    self.GroupNorm_0 = nn.GroupNorm(num_groups=min(in_ch // 4, 32), num_channels=in_ch, eps=1e-6)
    self.up = up
    self.down = down
    self.fir = fir
    self.fir_kernel = fir_kernel
    self.Conv_0 = conv3x3(in_ch, out_ch)
    if temb_dim is not None:
      self.Dense_0 = nn.Linear(temb_dim, out_ch)
      self.Dense_0.weight.data = default_init()(self.Dense_0.weight.shape)
      nn.init.zeros_(self.Dense_0.bias)

    self.GroupNorm_1 = nn.GroupNorm(num_groups=min(out_ch // 4, 32), num_channels=out_ch, eps=1e-6)
    self.Dropout_0 = nn.Dropout(dropout)
    self.Conv_1 = conv3x3(out_ch, out_ch, init_scale=init_scale)
    if in_ch != out_ch or up or down:
      self.Conv_2 = conv1x1(in_ch, out_ch)

    self.skip_rescale = skip_rescale
    self.act = act
    self.in_ch = in_ch
    self.out_ch = out_ch

  def forward(self, x, temb=None):
    h = self.act(self.GroupNorm_0(x))

    if self.up:
        h = naive_upsample(h)#F.interpolate(h, scale_factor=2, mode='nearest-exact')
        x = naive_upsample(x)#F.interpolate(x, scale_factor=2, mode='nearest-exact')
    elif self.down:
        h = naive_downsample(h)#F.avg_pool3d(h, kernel_size=2)
        x = naive_downsample(x)#F.avg_pool3d(x, kernel_size=2)


    h = self.Conv_0(h)
    # Add bias to each feature map conditioned on the time embedding
    if temb is not None:
      h += self.Dense_0(self.act(temb))[:, :, None, None, None]
    h = self.act(self.GroupNorm_1(h))
    h = self.Dropout_0(h)
    h = self.Conv_1(h)

    if self.in_ch != self.out_ch or self.up or self.down:
      x = self.Conv_2(x)

    if not self.skip_rescale:
      return x + h
    else:
      return (x + h) / np.sqrt(2.)
