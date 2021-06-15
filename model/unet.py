# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# from .modules.utils import DoubleConv, Down, Up, OutConv, initialize_weights
# from .modules.scse import sSE, cSE
# from .modules.attention_block import Attention_block
# from .modules.dblock import Dblock




# class Decoder(nn.Module):
#     def __init__(self, in_channels, out_channels, bilinear=True, scse=False, attention_block=False):
#         super(Decoder, self).__init__()
#         self.conv = DoubleConv(in_channels, out_channels)
#         self.scse = scse
#         if self.scse:
#             self.spatial_gate = sSE(out_channels)
#             self.channel_gate = cSE(out_channels)
#         self.attention_block = attention_block
#         if self.attention_block:
#             self.att = Attention_block(F_g=in_channels // 2,F_l=in_channels // 2,F_int=in_channels // 4)
#         if bilinear:
#             self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#         else:
#             self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)


#     def forward(self, x, e=None):
#         #  if self.bilinear:
#         #     x = F.upsample(x, scale_factor=2, mode='bilinear', align_corners=True)
#         # else:
#         #     self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
        
#         #print('x',x.size())
#         #print('e',e.size())
#         x = self.up(x)
           
#         if e is not None:
#             if self.attention_block:
#                  e = self.att(x,e)
#             x = torch.cat([x,e],1)

#         x = self.conv(x)
#         if self.scse:
#             #print('x_new',x.size())
#             g1 = self.spatial_gate(x)
#             #print('g1',g1.size())
#             g2 = self.channel_gate(x)
#             #print('g2',g2.size())
#             x = g1*x + g2*x
#         return x



# class Unet(nn.Module):
#     def __init__(self, n_channels, n_classes, bilinear, encode_dim=64, dropout_rate=0.5, scse=False, dblock=False, attention_block=False):
#         super(Unet, self).__init__()
#         self.inc = DoubleConv(n_channels, encode_dim)
#         self.down1 = Down(encode_dim, encode_dim*2)
#         self.down2 = Down(encode_dim*2, encode_dim*4)
#         self.down3 = Down(encode_dim*4, encode_dim*8)
#         self.down4 = Down(encode_dim*8, encode_dim*8)
#         self.use_dblock = dblock
#         if self.use_dblock:
#             self.dilate_center = Dblock(encode_dim*8, encode_dim*8)
#         # if not scse:
#         #     self.up1 = Up(encode_dim*16, encode_dim*4, bilinear)
#         #     self.up2 = Up(encode_dim*8, encode_dim*2, bilinear)
#         #     self.up3 = Up(encode_dim*4, encode_dim, bilinear)
#         #     self.up4 = Up(encode_dim*2, encode_dim, bilinear)
#         # else:
#         #     self.up1 = Decoder(encode_dim*16, encode_dim*4)
#         #     self.up2 = Decoder(encode_dim*8, encode_dim*2)
#         #     self.up3 = Decoder(encode_dim*4, encode_dim)
#         #     self.up4 = Decoder(encode_dim*2, encode_dim)
#         self.up1 = Decoder(encode_dim*16, encode_dim*4, bilinear=bilinear, scse=scse, attention_block=attention_block)
#         self.up2 = Decoder(encode_dim*8, encode_dim*2, bilinear=bilinear, scse=scse, attention_block=attention_block)
#         self.up3 = Decoder(encode_dim*4, encode_dim, bilinear=bilinear, scse=scse, attention_block=attention_block)
#         self.up4 = Decoder(encode_dim*2, encode_dim, bilinear=bilinear, scse=scse, attention_block=attention_block)
      
#         self.use_dropout = False
#         if dropout_rate > 0:
#             self.use_dropout = True
#             self.dropout = nn.Dropout2d(dropout_rate)
#         self.outc = OutConv(encode_dim, n_classes)
#         initialize_weights(self)

#     def forward(self, x):
#         x1 = self.inc(x)
#         x2 = self.down1(x1)
#         x3 = self.down2(x2)
#         x4 = self.down3(x3)
#         x5 = self.down4(x4)
#         if self.use_dblock:
#             x5 = self.dilate_center(x5)
#         x = self.up1(x5, x4)
#         x = self.up2(x, x3)
#         x = self.up3(x, x2)
#         x = self.up4(x, x1)
#         if self.use_dropout:
#             x = self.dropout(x)
#         x = self.outc(x)
#         return x



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

from .modules.utils import DoubleConv, Down, Up, OutConv, initialize_weights, conv_block, up_conv
from .modules.scse import sSE, cSE

from .modules.dblock import Dblock





# class Decoder(nn.Module):
#     def __init__(self, in_channels, out_channels, bilinear=True, scse=False, attention_block=False):
#         super(Decoder, self).__init__()
#         self.conv = DoubleConv(in_channels, out_channels)
#         self.scse = scse
#         if self.scse:
#             self.spatial_gate = sSE(out_channels)
#             self.channel_gate = cSE(out_channels)
#         self.attention_block = attention_block
#         if self.attention_block:
#             self.att = Attention_block(F_g=in_channels // 2,F_l=in_channels // 2,F_int=in_channels // 4)
#         if bilinear:
#             self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#         else:
#             self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)


#     def forward(self, x, e=None):
#         #  if self.bilinear:
#         #     x = F.upsample(x, scale_factor=2, mode='bilinear', align_corners=True)
#         # else:
#         #     self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
        
#         #print('x',x.size())
#         #print('e',e.size())
#         x = self.up(x)
           
#         if e is not None:
#             if self.attention_block:
#                  e = self.att(x,e)
#             x = torch.cat([x,e],1)

#         x = self.conv(x)
#         if self.scse:
#             #print('x_new',x.size())
#             g1 = self.spatial_gate(x)
#             #print('g1',g1.size())
#             g2 = self.channel_gate(x)
#             #print('g2',g2.size())
#             x = g1*x + g2*x
#         return x



# class Unet(nn.Module):
#     def __init__(self, n_channels, n_classes, bilinear, encode_dim=64, dropout_rate=0.5, scse=False, dblock=False, attention_block=False):
#         super(Unet, self).__init__()
#         self.inc = DoubleConv(n_channels, encode_dim)
#         self.down1 = Down(encode_dim, encode_dim*2)
#         self.down2 = Down(encode_dim*2, encode_dim*4)
#         self.down3 = Down(encode_dim*4, encode_dim*8)
#         self.down4 = Down(encode_dim*8, encode_dim*8)
#         self.use_dblock = dblock
#         if self.use_dblock:
#             self.dilate_center = Dblock(encode_dim*8, encode_dim*8)
#         # if not scse:
#         #     self.up1 = Up(encode_dim*16, encode_dim*4, bilinear)
#         #     self.up2 = Up(encode_dim*8, encode_dim*2, bilinear)
#         #     self.up3 = Up(encode_dim*4, encode_dim, bilinear)
#         #     self.up4 = Up(encode_dim*2, encode_dim, bilinear)
#         # else:
#         #     self.up1 = Decoder(encode_dim*16, encode_dim*4)
#         #     self.up2 = Decoder(encode_dim*8, encode_dim*2)
#         #     self.up3 = Decoder(encode_dim*4, encode_dim)
#         #     self.up4 = Decoder(encode_dim*2, encode_dim)
#         self.up1 = Decoder(encode_dim*16, encode_dim*4, bilinear=bilinear, scse=scse, attention_block=attention_block)
#         self.up2 = Decoder(encode_dim*8, encode_dim*2, bilinear=bilinear, scse=scse, attention_block=attention_block)
#         self.up3 = Decoder(encode_dim*4, encode_dim, bilinear=bilinear, scse=scse, attention_block=attention_block)
#         self.up4 = Decoder(encode_dim*2, encode_dim, bilinear=bilinear, scse=scse, attention_block=attention_block)
      
#         self.use_dropout = False
#         if dropout_rate > 0:
#             self.use_dropout = True
#             self.dropout = nn.Dropout2d(dropout_rate)
#         self.outc = OutConv(encode_dim, n_classes)
#         initialize_weights(self)

#     def forward(self, x):
#         x1 = self.inc(x)
#         x2 = self.down1(x1)
#         x3 = self.down2(x2)
#         x4 = self.down3(x3)
#         x5 = self.down4(x4)
#         if self.use_dblock:
#             x5 = self.dilate_center(x5)
#         x = self.up1(x5, x4)
#         x = self.up2(x, x3)
#         x = self.up3(x, x2)
#         x = self.up4(x, x1)
#         if self.use_dropout:
#             x = self.dropout(x)
#         x = self.outc(x)
#         return x



class Unet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear, encode_dim=64, dropout_rate=0.5, scse=False, dblock=False, attention_block=False, rrcnn_block=False, rrcnn_block_t=2):
        super(Unet,self).__init__()
        
        self.use_scse = scse
        self.use_dblock = dblock
        self.use_attention_block = attention_block
        self.use_rrcnn_block = rrcnn_block
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        

        self.Conv1 = conv_block(ch_in=n_channels,ch_out=encode_dim)
        self.Conv2 = conv_block(ch_in=encode_dim,ch_out=encode_dim*2)
        self.Conv3 = conv_block(ch_in=encode_dim*2,ch_out=encode_dim*4)
        self.Conv4 = conv_block(ch_in=encode_dim*4,ch_out=encode_dim*8)
        self.Conv5 = conv_block(ch_in=encode_dim*8,ch_out=encode_dim*16)

        self.Up_conv5 = conv_block(ch_in=encode_dim*16, ch_out=encode_dim*8)
        self.Up_conv4 = conv_block(ch_in=encode_dim*8, ch_out=encode_dim*4)
        self.Up_conv3 = conv_block(ch_in=encode_dim*4, ch_out=encode_dim*2)
        self.Up_conv2 = conv_block(ch_in=encode_dim*2, ch_out=encode_dim)
        
        if self.use_dblock:
            self.dilate_center = Dblock(encode_dim*16, encode_dim*16)

        self.Up5 = up_conv(ch_in=encode_dim*16,ch_out=encode_dim*8)

        
        if self.use_scse:
            self.spatial_gate5 = sSE(encode_dim*8)
            self.channel_gate5 = cSE(encode_dim*8)
        
        self.Up4 = up_conv(ch_in=encode_dim*8,ch_out=encode_dim*4)

        
        if self.use_scse:
            self.spatial_gate4 = sSE(encode_dim*4)
            self.channel_gate4 = cSE(encode_dim*4)
        
        self.Up3 = up_conv(ch_in=encode_dim*4,ch_out=encode_dim*2)

        
        if self.use_scse:
            self.spatial_gate3 = sSE(encode_dim*2)
            self.channel_gate3 = cSE(encode_dim*2)
        
        self.Up2 = up_conv(ch_in=encode_dim*2,ch_out=encode_dim)

        
        if self.use_scse:
            self.spatial_gate2 = sSE(encode_dim)
            self.channel_gate2 = cSE(encode_dim)

        self.Conv_1x1 = nn.Conv2d(encode_dim,n_classes,kernel_size=1,stride=1,padding=0)


    def forward(self,x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)
        
        if self.use_dblock:
            x5  = self.dilate_center(x5)

        # decoding + concat path
        d5 = self.Up5(x5)

        d5 = torch.cat((x4,d5),dim=1)
        d5 = self.Up_conv5(d5)
        if self.use_scse:
            g1_5 = self.spatial_gate5(d5)
            #print('g1',g1.size())
            g2_5 = self.channel_gate5(d5)
            #print('g2',g2.size())
            d5 = g1_5*d5 + g2_5*d5
        
        d4 = self.Up4(d5)

        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)
        if self.use_scse:
            g1_4 = self.spatial_gate4(d4)
            #print('g1',g1.size())
            g2_4 = self.channel_gate4(d4)
            #print('g2',g2.size())
            d4 = g1_4*d4 + g2_4*d4

        d3 = self.Up3(d4)

        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)
        if self.use_scse:
            g1_3 = self.spatial_gate3(d3)
            #print('g1',g1.size())
            g2_3 = self.channel_gate3(d3)
            #print('g2',g2.size())
            d3 = g1_3*d3 + g2_3*d3

        d2 = self.Up2(d3)

        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)
        if self.use_scse:
            g1_2 = self.spatial_gate2(d2)
            #print('g1',g1.size())
            g2_2 = self.channel_gate2(d2)
            #print('g2',g2.size())
            d2 = g1_2*d2 + g2_2*d2

        d1 = self.Conv_1x1(d2)

        return d1
