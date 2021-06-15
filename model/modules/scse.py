import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial

nonlinearity = partial(F.relu,inplace=True)

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
    def __init__(self, out_channels, act_fun=nonlinearity):
        super(cSE, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels,out_channels=int(out_channels/2),kernel_size=1,padding=0),
            nn.BatchNorm2d(int(out_channels/2))
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=int(out_channels/2),out_channels=out_channels,kernel_size=1,padding=0),
            nn.BatchNorm2d(out_channels)
        )
        self.act_fun = act_fun
    def forward(self,x):
        x=nn.AvgPool2d(x.size()[2:])(x)
        #print('channel',x.size())
        x=self.conv1(x)
        x=self.act_fun(x)
        x=self.conv2(x)
        x=F.sigmoid(x)
        return x