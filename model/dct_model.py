# -*- coding: utf-8 -*-
"""
dct-Net 네트워크 
사이드 프로젝트 필요 : 
() 1. kaist 선행 연구 : https://github.com/plok5308/DJPEG-torch/blob/ab19c3d7ee9fa8fa10d7b40e77cead1410897d5c/djpegnet.py
(0) 2. kaist CAT-Net : https://github.com/mjkwon2021/CAT-Net/blob/90739212417fe78b6bc7bc3b3a5fd93902fa67b1/lib/models/network_CAT.py#L315
() 3.
_crop_size = (256,256)
_grid_crop = True
_blocks = ['RGB', 'DCTcoef', 'DCTvol', 'qtable']
tamp_list = None
DCT_channels = 1
"""

import torch
import torch.nn as nn
from torch.nn import BatchNorm2d


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = BatchNorm2d(planes, momentum=0.01)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = BatchNorm2d(planes, momentum=0.01)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

# CAT_Net module
class DCT_Learning_Module(nn.Module):
    def __init__(self):
        super(DCT_Learning_Module,self).__init__()
        self.dc_layer0_dil = nn.Sequential(
            nn.Conv2d(in_channels=21,
                      out_channels=64,
                      kernel_size=3,
                      stride=1,
                      dilation=8,
                      padding=8),
            nn.BatchNorm2d(64, momentum=0.01),
            nn.ReLU(inplace=True)
        )
        self.dc_layer1_tail = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=4, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(4, momentum=0.01),
            nn.ReLU(inplace=True)
        )
      
    def forward(self, x, qtable):
        RGB, DCTcoef = x[:, :3, :, :], x[:, 3:, :, :]
        # DCT Stream
        x = self.dc_layer0_dil(DCTcoef)
        x = self.dc_layer1_tail(x)
        B, C, H, W = x.shape
        x0 = x.reshape(B, C, H // 8, 8, W // 8, 8).permute(0, 1, 3, 5, 2, 4).reshape(B, 64 * C, H // 8,
                                                                                     W // 8)  # [B, 256, 32, 32]
        x_temp = x.reshape(B, C, H // 8, 8, W // 8, 8).permute(0, 1, 3, 5, 2, 4)  # [B, C, 8, 8, 32, 32]
        q_temp = qtable.unsqueeze(-1).unsqueeze(-1)  # [B, 1, 8, 8, 1, 1]
        xq_temp = x_temp * q_temp  # [B, C, 8, 8, 32, 32]
        x1 = xq_temp.reshape(B, 64 * C, H // 8, W // 8)  # [B, 256, 32, 32]
        x = torch.cat([x0, x1], dim=1)
        return x


from model.unet_parts import *
class FC_DCT_Only(nn.Module):
    def __init__(self,n_channels=512,n_classes=1):
        super(FC_DCT_Only,self).__init__() 
        # (B,512,64,64)
        self.dct_learning_module = DCT_Learning_Module()
        self.down = RRU_first_down(512, 256)
        self.down1 = RRU_down(256, 256) # 32
        self.down2 = RRU_down(256, 64) # 16
        self.down3 = RRU_down(64, 32) # 8
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32*8*8,512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512,n_classes)
        self.relu = nn.ReLU()

    def forward(self,x,qtable): # 일단 jpeg 학습 모듈 ouput이 (B,512,64,64)
        x = self.dct_learning_module(x,qtable)
        x = self.down(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        

        return x