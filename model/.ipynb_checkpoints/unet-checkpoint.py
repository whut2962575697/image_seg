# -*- encoding: utf-8 -*-
'''
@File    :   unet.py    
@Contact :   whut.hexin@foxmail.com
@License :   (C)Copyright 2017-2018, HeXin

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2020/1/28 13:45   xin      1.0         None
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules.utils import DoubleConv, Down, Up, OutConv, initialize_weights



class sSE(nn.Module):
    def __init__(self, out_channels):
        super(sSE, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=out_channels,out_channels=1,kernel_size=1,padding=0),
            nn.BatchNorm2d(1))
    def forward(self,x):
        x=self.conv(x)
        #print('spatial',x.size())
        x=F.sigmoid(x)
        return x

class cSE(nn.Module):
    def __init__(self, out_channels):
        super(cSE, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels,out_channels=int(out_channels/2),kernel_size=1,padding=0),
            nn.BatchNorm2d(int(out_channels/2))
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=int(out_channels/2),out_channels=out_channels,kernel_size=1,padding=0),
            nn.BatchNorm2d(out_channels)
        )
    def forward(self,x):
        x=nn.AvgPool2d(x.size()[2:])(x)
        #print('channel',x.size())
        x=self.conv1(x)
        x=F.relu(x)
        x=self.conv2(x)
        x=F.sigmoid(x)
        return x



class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.spatial_gate = sSE(out_channels)
        self.channel_gate = cSE(out_channels)
        # self.fam = FAM_Block(in_channels // 2)
        # self.aam = Augmented_Attention_Module(in_channels // 2, in_channels // 2)


    def forward(self, x, e=None):
        # x = self.fam(x)
        x = F.upsample(x, scale_factor=2, mode='bilinear', align_corners=True)
        #print('x',x.size())
        #print('e',e.size())
        if e is not None:
            # e = self.aam(e, x)
            x = torch.cat([x,e],1)

        x = self.conv(x)
        #print('x_new',x.size())
        g1 = self.spatial_gate(x)
        #print('g1',g1.size())
        g2 = self.channel_gate(x)
        #print('g2',g2.size())
        x = g1*x + g2*x
        return x


class Augmented_Attention_Module(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.c1_conv1 = nn.Sequential(nn.Conv2d(self.c1, self.c2, 1),
                                      nn.BatchNorm2d(self.c2))

        self.c1_conv2 = nn.Sequential(nn.Conv2d(self.c1, self.c2, 1),
                                      nn.BatchNorm2d(self.c2),
                                      nn.ReLU())

        self.c2_conv1 = nn.Sequential(nn.Conv2d(self.c2, self.c2, 1),
                                      nn.BatchNorm2d(self.c2),
                                      nn.ReLU()
                                      )
        self.merge_conv = nn.Sequential(nn.Conv2d(self.c2, self.c2, 1),
                                        nn.BatchNorm2d(self.c2))

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x1, x2):
        x1_1 = self.c1_conv1(x1)
        c1_v = self.gap(x1)
        c1_v = self.c1_conv2(c1_v)
        c2_v = self.gap(x2)
        c2_v = self.c2_conv1(c2_v)
        merge_v = self.merge_conv(c1_v+c2_v)
        att_v = self.softmax(merge_v)
        att_x1 = att_v*x1_1
        return att_x1


class FAM_Block(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.down1 = nn.AvgPool2d(8)
        self.down2 = nn.AvgPool2d(4)
        self.down3 = nn.AvgPool2d(2)
        self.conv1 = nn.Conv2d(feature_dim, feature_dim, 3, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(feature_dim, feature_dim, 3, 1, 1, bias=False)
        self.conv3 = nn.Conv2d(feature_dim, feature_dim, 3, 1, 1, bias=False)
        self.relu = nn.ReLU()
        self.merge_conv = nn.Conv2d(feature_dim, feature_dim, 3, 1, 1, bias=False)

    def forward(self, x):
        x_size = x.size()
        x1 = self.conv1(self.down1(x))
        x1 = F.interpolate(x1, x_size[2:], mode='bilinear', align_corners=True)
        res = torch.add(x, x1)
        x2 = self.conv2(self.down2(x))
        x2 = F.interpolate(x2, x_size[2:], mode='bilinear', align_corners=True)
        res = torch.add(res, x2)
        x3 = self.conv3(self.down3(x))
        x3 = F.interpolate(x3, x_size[2:], mode='bilinear', align_corners=True)
        res = torch.add(res, x3)
        res = self.relu(res)
        res = self.merge_conv(res)
        return res


class Unet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear, encode_dim=64, dropout_rate=0.5):
        super(Unet, self).__init__()
        self.inc = DoubleConv(n_channels, encode_dim)
        self.down1 = Down(encode_dim, encode_dim*2)
        self.down2 = Down(encode_dim*2, encode_dim*4)
        self.down3 = Down(encode_dim*4, encode_dim*8)
        self.down4 = Down(encode_dim*8, encode_dim*8)

        self.up1 = Up(encode_dim*16, encode_dim*4, bilinear)
        self.up2 = Up(encode_dim*8, encode_dim*2, bilinear)
        self.up3 = Up(encode_dim*4, encode_dim, bilinear)
        self.up4 = Up(encode_dim*2, encode_dim, bilinear)
        # self.up1 = Decoder(encode_dim * 16, encode_dim * 4)
        # self.up2 = Decoder(encode_dim * 8, encode_dim * 2)
        # self.up3 = Decoder(encode_dim * 4, encode_dim)
        # self.up4 = Decoder(encode_dim * 2, encode_dim)
        self.use_dropout = False
        if dropout_rate > 0:
            self.use_dropout = True
            self.dropout = nn.Dropout2d(dropout_rate)
        self.outc = OutConv(encode_dim, n_classes)
        initialize_weights(self)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        if self.use_dropout:
            x = self.dropout(x)
        x = self.outc(x)
        return x





