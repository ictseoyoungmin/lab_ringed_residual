# -*- coding: utf-8 -*-
"""
RRU-Net 네트워크 body
"""
# from unet.unet_parts import *
from .unet_parts import *
import torch.nn as nn

class Ringed_Res_Unet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1):
        super(Ringed_Res_Unet, self).__init__()
        self.down = RRU_first_down(n_channels, 32)
        self.down1 = RRU_down(32, 64)
        self.down2 = RRU_down(64, 128)
        self.down3 = RRU_down(128, 256)
        self.down4 = RRU_down(256, 256)
        self.up1 = RRU_up(512, 128)
        self.up2 = RRU_up(256, 64)
        self.up3 = RRU_up(128, 32)
        self.up4 = RRU_up(64, 32)
        self.out = outconv(32, n_classes)
        # self.classifier = Classisifer(512*512,2) # 이미지 사이즈 512, outconv ch 1 기준

    def forward(self, x):
        x1 = self.down(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x_seg = self.out(x) # (1,H,W)
        # x = x_seg.view(-1)
        # x_class = self.classifier(x) # [1,0] : 정상, [0,1] : 위조 
        return x_seg

# model = Ringed_Res_Unet(3,1)
# print('# generator parameters:', 1.0 * sum(param.numel() for param in model.parameters())/1000000)
# params = list(model.named_parameters())
# for i in range(len(params)):
#     (name, param) = params[i]
# print(name)
# print(param.shape)