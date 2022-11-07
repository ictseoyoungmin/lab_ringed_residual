# -*- coding: utf-8 -*-
"""
dct-Net 네트워크 
사이드 프로젝트 필요 : 
() 1. kaist 선행 연구 : https://github.com/plok5308/DJPEG-torch/blob/ab19c3d7ee9fa8fa10d7b40e77cead1410897d5c/djpegnet.py
(0) 2. kaist CAT-Net : https://github.com/mjkwon2021/CAT-Net/blob/90739212417fe78b6bc7bc3b3a5fd93902fa67b1/lib/models/network_CAT.py#L315
() 3.
"""

import torch
import torch.nn as nn

class DCTNet(nn.Module):
    def __init__(self):
        super(DCTNet,self).__init__()

      
    def forward(self,DCTcoef,qtable):
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
        x = self.dc_layer2(x)  # x.shape = torch.Size([1, 96, 64, 64])


class DCT_layer0(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DCT_layer0, self).__init__()
        self.conv = nn.Sequential(
             nn.Conv2d(in_channels=21,
                      out_channels=64,
                      kernel_size=3,
                      stride=1,
                      dilation=8,
                      padding=8),
            nn.GroupNorm(32, out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class DCT_layer1(nn.Module):
    def __init__(self, in_ch=64, out_ch=4):
        super(DCT_layer1, self).__init__()
        self.conv =nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, stride=1, padding=0, bias=False),
            nn.GroupNorm(32, out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x
