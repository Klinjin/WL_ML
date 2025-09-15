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

####### TOO BIG for RAM (~20 GB) 

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
        self.num_res_blocks = num_res_blocks = 1
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
    
        self._feature_size = in_ch * height//4 * width//4

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



    def forward(self, x): #cond(5,)-->MLP score(N,)=weight+bias
        x = self.maybe_concat_fourier(x)
        
        modules = self.all_modules
        m_idx = 0
        
        # Downsampling block
        hs = [modules[m_idx](x)]
        m_idx += 1
        for i_level in range(self.num_resolutions):
          # Residual blocks for this resolution
          for i_block in range(self.num_res_blocks):
            h = modules[m_idx](hs[-1])
            if isinstance(modules[m_idx], ResnetBlockBigGAN):
                channels = h.shape[1]
            m_idx += 1
            hs.append(h)
    
          if i_level != self.num_resolutions - 1:
            h = modules[m_idx](hs[-1])
            if isinstance(modules[m_idx], ResnetBlockBigGAN):
                channels = h.shape[1]
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


import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample:
            residual = self.downsample(x)
            
        out += residual
        out = self.relu(out)
        
        return out

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(x_cat)
        return self.sigmoid(out)

class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention()
        
    def forward(self, x):
        out = x * self.channel_attention(x)
        out = out * self.spatial_attention(out)
        return out

class ResNetWithAttention(nn.Module):
    def __init__(self, height, width, num_targets):
        super(ResNetWithAttention, self).__init__()
        
        # Initial convolution
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residual layers
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        # Attention modules
        self.attention1 = CBAM(64)
        self.attention2 = CBAM(128)
        self.attention3 = CBAM(256)
        self.attention4 = CBAM(512)
        
        # Calculate feature size after convolutions
        self._feature_size = self._get_conv_output_size(height, width)
        
        # Fully connected layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_stack = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_targets)
        )
        
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride, downsample))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
            
        return nn.Sequential(*layers)
    
    def _get_conv_output_size(self, height, width):
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, height, width)
            x = self.conv1(dummy_input)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.avgpool(x)
            
            return int(np.prod(x.size()))
    
    def forward(self, x):
        # Initial convolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # Residual blocks with attention
        x = self.layer1(x)
        x = self.attention1(x)
        
        x = self.layer2(x)
        x = self.attention2(x)
        
        x = self.layer3(x)
        x = self.attention3(x)
        
        x = self.layer4(x)
        x = self.attention4(x)
        
        # Global average pooling and classification
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc_stack(x)
        
        means = x[:, :2]
        log_sigmas = x[:, 2:]
        sigmas = torch.exp(log_sigmas)
        
        return means, sigmas