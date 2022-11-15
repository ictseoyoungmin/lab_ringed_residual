# -*- coding: utf-8 -*-
"""
RRU-Net 네트워크 body
"""
# from unet.unet_parts import *
from .unet_parts import *
from .dct_model import *
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

class DCT_RRUnet(nn.Module):
    def __init__(self,n_channals=3, n_classes=1):
        super(DCT_RRUnet, self).__init__()
         # RGB
        self.down = RRU_first_down(n_channals, 32) # size 512
        self.down1 = RRU_down(32, 64) # 512->256
        self.down2 = RRU_down(64, 128) # 256->128
        self.down3 = RRU_down(128, 256) # 128->64
        self.down4 = RRU_down(256, 256) # 64->32
        self.up1 = RRU_up_dct(512, 128) # 32->64
        self.up2 = RRU_up_dct(256, 64) # 64->128
        self.up3 = RRU_up(128, 32) # 128->256
        self.up4 = RRU_up(64, 32) # 256->512
        self.out = outconv(32, n_classes) # 512

        # jpeg compressed artifact learning layer
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
        # DCT 가정 1. stacked model / 2. u model
        self.down_dct = RRU_first_down(512, 128) # size 64
        self.down1_dct = RRU_down(128, 256) # 64->32
        # self.down2_dct = RRU_down(64, 128) # 32->16

       
        # self.classifier = Classisifer(512*512,2) # 이미지 사이즈 512, outconv ch 1 기준

    def forward(self, x,qtable):
        RGB, DCTcoef = x[:, :3, :, :], x[:, 3:, :, :]

        # RGB down 
        x01 = self.down(RGB) # 512
        x02 = self.down1(x01) # 256
        x03 = self.down2(x02) # 128
        x04 = self.down3(x03) # 64
        x05 = self.down4(x04) # 32

        # DCT down
        x = self.dc_layer0_dil(DCTcoef)
        x = self.dc_layer1_tail(x)
        B, C, H, W = x.shape
        x0 = x.reshape(B, C, H // 8, 8, W // 8, 8).permute(0, 1, 3, 5, 2, 4).reshape(B, 64 * C, H // 8, W // 8)  # [B, 256, 32, 32]
        x_temp = x.reshape(B, C, H // 8, 8, W // 8, 8).permute(0, 1, 3, 5, 2, 4)  # [B, C, 8, 8, 32, 32]
        q_temp = qtable.unsqueeze(-1).unsqueeze(-1)  # [B, 1, 8, 8, 1, 1]
        xq_temp = x_temp * q_temp  # [B, C, 8, 8, 32, 32]
        x1 = xq_temp.reshape(B, 64 * C, H // 8, W // 8)  # [B, 256, 32, 32]
        x = torch.cat([x0, x1], dim=1) #[B, 512,64,64]
        
        x10 = self.down_dct(x) # 64 
        x20 = self.down1_dct(x10) # 64->32

        x = self.up1(x05, x04, x20)
        x = self.up2(x, x03, x10)
        x = self.up3(x, x02)
        x = self.up4(x, x01)
        x_seg = self.out(x) # (1,H,W)
      
        return x_seg


if __name__ == '__main__':
    print('as')
    