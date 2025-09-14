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

import layerspp
from layerspp import *
import torch.nn as nn
import functools
import torch
import numpy as np


ResnetBlockBigGAN = layerspp.ResnetBlockBigGANpp
SelfAttentionBlock = layerspp.SelfAttentionBlock
conv3x3 = layerspp.conv3x3
conv1x1 = layerspp.conv1x1

default_initializer = layers.default_init



class BigGANUNet2DModel(nn.Module):
    def __init__(self, height, width, n_classes, n_channels=1, ch_mult = (1, 2, 1), bilinear: bool = False, use_fourier_features: bool = False, attention: bool = False):
        super(BigGANUNet2DModel, self).__init__()
        self.use_fourier_features = use_fourier_features
        self.attention = attention
        if use_fourier_features:
            self.fourier_features = FourierFeatures(
                first=-2.0,
                last=1,
                step=1,
            )
        if use_fourier_features:
            n_channels *= 1 + self.fourier_features.num_features
        nf = 32
        self.ch_mult = ch_mult 
        self.num_res_blocks = num_res_blocks = 3
        self.num_resolutions =num_resolutions= len(ch_mult)
        self.act = nn.SiLU()
        
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        AttnBlock = functools.partial(SelfAttentionBlock, init_scale=1e-2, dimensions=2)
        ResnetBlock = functools.partial(ResnetBlockBigGAN,
                                    act=nn.LeakyReLU(0.1), #Ramanah 2020 super-res
                                    dropout=0.2,
                                    fir=False,
                                    fir_kernel=[1, 3, 3, 1],
                                    init_scale=0,
                                    skip_rescale=True,
                                    temb_dim=None)
        
        modules = []
        input_channels = n_channels
        output_channels =n_classes

        modules.append(conv3x3(input_channels, nf))
        hs_c = [nf]
    
        in_ch = nf
        for i_level in range(num_resolutions):
          # Residual blocks for this resolution
          for i_block in range(num_res_blocks):
            out_ch = nf * ch_mult[i_level]
            modules.append(ResnetBlock(in_ch=in_ch, out_ch=out_ch))
            in_ch = out_ch
            hs_c.append(in_ch)
    
          if i_level != num_resolutions - 1:
            modules.append(ResnetBlock(down=True,in_ch=in_ch))
            hs_c.append(in_ch)
    
        in_ch = hs_c[-1]
        modules.append(ResnetBlock(in_ch=in_ch))
        if self.attention:
            modules.append(AttnBlock(channels=in_ch))
        modules.append(ResnetBlock(in_ch=in_ch))
    
        self._feature_size = in_ch * height * width

        self.fc_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self._feature_size, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, output_channels)
        )
        self.all_modules = nn.ModuleList(modules)
        self.double()



    def forward(self, x, cond=None): #cond(5,)-->MLP score(N,)=weight+bias
        param = self.MLP(cond)
        x = self.maybe_concat_fourier(x)
        x = x.double()
        
        modules = self.all_modules
        m_idx = 0
        param_idx = 0
        
        # Downsampling block
        hs = [modules[m_idx](x)]
        m_idx += 1
        for i_level in range(self.num_resolutions):
          # Residual blocks for this resolution
          for i_block in range(self.num_res_blocks):
            h = modules[m_idx](hs[-1])
            if isinstance(modules[m_idx], ResnetBlockBigGAN):
                channels = h.shape[1]
                h = self.film(h, param[:, param_idx:param_idx + channels * 2])
                param_idx += channels * 2
            m_idx += 1
            hs.append(h)
    
          if i_level != self.num_resolutions - 1:
            h = modules[m_idx](hs[-1])
            if isinstance(modules[m_idx], ResnetBlockBigGAN):
                channels = h.shape[1]
                h = self.film(h, param[:, param_idx:param_idx + channels * 2])
                param_idx += channels * 2
            m_idx += 1
            hs.append(h)
    
        h = hs[-1]
        h = modules[m_idx](h)
        m_idx += 1
        if self.attention:
            h = modules[m_idx](h)
            m_idx += 1
        h = modules[m_idx](h)
    
        assert not hs

        self.fc_stack(h)
    
        assert m_idx == len(modules)
        
        #assert h.dtype == torch.float64, "Output is not in double precision"
        means = h[:, :2]
        log_sigmas = h[:, 2:]    # Predict log(σ) to ensure positivity
        sigmas = torch.exp(log_sigmas)
        return means, sigmas        


    def maybe_concat_fourier(self, x):
        if self.use_fourier_features:
            return torch.cat([x, self.fourier_features(x)], dim=1)
        return x
    
    def use_checkpointing(self):
        for i, module in enumerate(self.all_modules):
            self.all_modules[i] = torch.utils.checkpoint(module)

