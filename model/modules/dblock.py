from __future__ import print_function
from __future__ import division
from torch.nn import Module,Conv2d,ConvTranspose2d,functional
from functools import partial
import torch.nn as nn
import torch.nn.functional as F
import torch

from .scse import cSE

nonlinearity = partial(functional.relu, inplace=True)
# class Dblock(Module):
#     def __init__(self, input_chs,output_chs):
#         super(Dblock, self).__init__()
#         self.nonlinearity = partial(functional.relu, inplace=True)
#         self.dilate1 = Conv2d(input_chs,output_chs, kernel_size=3, dilation=1, padding=1)
#         self.dilate2 = Conv2d(input_chs,output_chs, kernel_size=3, dilation=2, padding=2)
#         self.dilate3 = Conv2d(input_chs,output_chs, kernel_size=3, dilation=4, padding=4)
#         self.dilate4 = Conv2d(input_chs,output_chs, kernel_size=3, dilation=8, padding=8)
#         self.dilate5 = Conv2d(input_chs,output_chs, kernel_size=3, dilation=16, padding=16)
#         self.dilate6 = Conv2d(input_chs,output_chs, kernel_size=3, dilation=32, padding=32)
#         for m in self.modules():
#             if isinstance(m, Conv2d) or isinstance(m, ConvTranspose2d):
#                 if m.bias is not None:
#                     m.bias.data.zero_()

#     def forward(self, x):
#         dilate1_out = self.nonlinearity(self.dilate1(x))
#         dilate2_out = self.nonlinearity(self.dilate2(dilate1_out))
#         dilate3_out = self.nonlinearity(self.dilate3(dilate2_out))
#         dilate4_out = self.nonlinearity(self.dilate4(dilate3_out))
#         dilate5_out = self.nonlinearity(self.dilate5(dilate4_out))
#         dilate6_out = self.nonlinearity(self.dilate6(dilate5_out))
#         out = dilate1_out \
#               + dilate2_out \
#               + dilate3_out \
#               + dilate4_out \
#               + dilate5_out \
#               + dilate6_out
#         return out


# class Dblock(nn.Module):
#     def __init__(self,channel, out_channel, act_fn=nonlinearity):
#         super(Dblock, self).__init__()
#         self.dilate1 = nn.Conv2d(channel, out_channel, kernel_size=3, dilation=1, padding=1)
#         self.dilate2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, dilation=2, padding=2)
#         self.dilate3 = nn.Conv2d(out_channel, out_channel, kernel_size=3, dilation=4, padding=4)
#         self.dilate4 = nn.Conv2d(out_channel, out_channel, kernel_size=3, dilation=8, padding=8)
#         # self.dilate5 = nn.Conv2d(out_channel, out_channel, kernel_size=3, dilation=16, padding=16)
#         self.act_fn = act_fn
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
#                 if m.bias is not None:
#                     m.bias.data.zero_()
                    
#     def forward(self, x):
#         dilate1_out = self.act_fn(self.dilate1(x))
#         dilate2_out = self.act_fn(self.dilate2(dilate1_out))
#         dilate3_out = self.act_fn(self.dilate3(dilate2_out))
#         dilate4_out = self.act_fn(self.dilate4(dilate3_out))
#         # dilate5_out = self.act_fn(self.dilate5(dilate4_out))
#         out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out
        # return out


class Dblock(nn.Module):
    def __init__(self, in_channel, out_channel, act_fn=nn.ReLU(inplace=True)):
        super(Dblock, self).__init__()
        self.conv = nn.Conv2d(in_channel*5, out_channel, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channel)
        self.act_fn = act_fn
        self.channel_gate = cSE(in_channel*5, act_fn)
        self.dilate1 = nn.Conv2d(in_channel, in_channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(in_channel, in_channel, kernel_size=3, dilation=2, padding=2)
        self.dilate3 = nn.Conv2d(in_channel, in_channel, kernel_size=3, dilation=4, padding=4)
        self.dilate4 = nn.Conv2d(in_channel, in_channel, kernel_size=3, dilation=8, padding=8)
        # self.dilate5 = nn.Conv2d(channel, channel, kernel_size=3, dilation=16, padding=16)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    # def forward(self, x):
    #     dilate1_out = nonlinearity(self.dilate1(x))
    #     dilate2_out = nonlinearity(self.dilate2(dilate1_out))
    #     dilate3_out = nonlinearity(self.dilate3(dilate2_out))
    #     dilate4_out = nonlinearity(self.dilate4(dilate3_out))
    #     # dilate5_out = nonlinearity(self.dilate5(dilate4_out))
    #     out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out  # + dilate5_out
    #     return out
    def forward(self, x):
        dilate1_out = self.act_fn(self.dilate1(x))
        dilate2_out = self.act_fn(self.dilate2(x))
        dilate3_out = self.act_fn(self.dilate3(x))
        dilate4_out = self.act_fn(self.dilate4(x))
        # dilate5_out = nonlinearity(self.dilate5(dilate4_out))
        # out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out
        # out = (x + dilate1_out + dilate2_out + dilate3_out + dilate4_out)/5.0  # + dilate5_out
        out = torch.cat((x, dilate1_out, dilate2_out, dilate3_out, dilate4_out), dim=1)
        out = self.channel_gate(out)*out
        out = self.conv(out)
        out = self.bn(out)
        return out


class SPPblock(nn.Module):
    def __init__(self, in_channels):
        super(SPPblock, self).__init__()
        self.pool1 = nn.MaxPool2d(kernel_size=[2, 2], stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=[3, 3], stride=3)
        self.pool3 = nn.MaxPool2d(kernel_size=[5, 5], stride=5)
        self.pool4 = nn.MaxPool2d(kernel_size=[6, 6], stride=6)

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=1, padding=0)

    def forward(self, x):
        self.in_channels, h, w = x.size(1), x.size(2), x.size(3)
        self.layer1 = F.upsample(self.conv(self.pool1(x)), size=(h, w), mode='bilinear')
        self.layer2 = F.upsample(self.conv(self.pool2(x)), size=(h, w), mode='bilinear')
        self.layer3 = F.upsample(self.conv(self.pool3(x)), size=(h, w), mode='bilinear')
        self.layer4 = F.upsample(self.conv(self.pool4(x)), size=(h, w), mode='bilinear')

        out = torch.cat([self.layer1, self.layer2, self.layer3, self.layer4, x], 1)

        return out