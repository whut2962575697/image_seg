"""
Codes of LinkNet based on https://github.com/snakers4/spacenet-three
"""
import torch
import torch.nn as nn
from torch.autograd import Variable
from .modules.backbones.resnet import resnet18, resnet34, resnet50, resnet101
from .modules.backbones.resnest import resnest50, resnest101
from .modules.scse import sSE, cSE
from .modules.mish import Mish

from .modules.cc import CC_module as CrissCrossAttention
import torch.nn.functional as F

from functools import partial

nonlinearity = partial(F.relu,inplace=True)


class RCCAModule(nn.Module):
    def __init__(self, in_channels, out_channels, act_fn=nn.ReLU(inplace=True)):
        super(RCCAModule, self).__init__()
        inter_channels = in_channels // 4
        self.conva = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(inter_channels),act_fn)
        self.cca = CrissCrossAttention(inter_channels)
        self.convb = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(inter_channels),act_fn)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels+inter_channels, out_channels, kernel_size=3, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(out_channels),act_fn,
            nn.Dropout2d(0.1),
            # nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
            )

    def forward(self, x, recurrence=2):
        output = self.conva(x)
        for i in range(recurrence):
            output = self.cca(output)
        output = self.convb(output)

        output = self.bottleneck(torch.cat([x, output], 1))
        return output


class Dblock_more_dilate(nn.Module):
    def __init__(self,channel):
        super(Dblock_more_dilate, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=2, padding=2)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=4, padding=4)
        self.dilate4 = nn.Conv2d(channel, channel, kernel_size=3, dilation=8, padding=8)
        self.dilate5 = nn.Conv2d(channel, channel, kernel_size=3, dilation=16, padding=16)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()
                    
    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.dilate2(dilate1_out))
        dilate3_out = nonlinearity(self.dilate3(dilate2_out))
        dilate4_out = nonlinearity(self.dilate4(dilate3_out))
        dilate5_out = nonlinearity(self.dilate5(dilate4_out))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out + dilate5_out
        return out


class Dblock(nn.Module):
    def __init__(self,channel):
        super(Dblock, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=2, padding=2)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=4, padding=4)
        self.dilate4 = nn.Conv2d(channel, channel, kernel_size=3, dilation=8, padding=8)
        #self.dilate5 = nn.Conv2d(channel, channel, kernel_size=3, dilation=16, padding=16)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()
                    
    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.dilate2(dilate1_out))
        dilate3_out = nonlinearity(self.dilate3(dilate2_out))
        dilate4_out = nonlinearity(self.dilate4(dilate3_out))
        #dilate5_out = nonlinearity(self.dilate5(dilate4_out))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out# + dilate5_out
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


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters, scse=False, act_fn=nonlinearity):
        super(DecoderBlock,self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)



        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = act_fn

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)


        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = act_fn

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)


        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = act_fn

        self.use_scse = scse
        if self.use_scse:
            self.spatial_gate = sSE(in_channels // 4)
            self.channel_gate = cSE(in_channels // 4, act_fun=act_fn)

    def forward(self, x):
        x = self.conv1(x)

        x = self.norm1(x)
        x = self.relu1(x)
        f = x
        if self.use_scse:
            g1 = self.spatial_gate(x)
            #print('g1',g1.size())
            g2 = self.channel_gate(x)
            #print('g2',g2.size())
            x = g1*x + g2*x

            # concat_input
            x = x + f

        x = self.deconv2(x)

        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)

        x = self.norm3(x)
        x = self.relu3(x)
            
        return x


class ConvBn2d(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=(3, 3), stride=(1, 1),
                 padding=(1, 1), groups=1, dilation=1, norm_layer=nn.BatchNorm2d):
        super(ConvBn2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              bias=False,
                              groups=groups,
                              dilation=dilation)
        self.bn = norm_layer(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class CenterBlock(nn.Module):
    def __init__(self, in_channels, out_channels, scse=False, use_mish=False, norm_layer=nn.BatchNorm2d):
        super(CenterBlock, self).__init__()
        self.scse = scse
        if use_mish:
            self.act_fn = Mish()
        else:
            self.act_fn = nn.ReLU(inplace=True)
        if self.scse:
            self.spatial_gate = sSE(out_channels)
            self.channel_gate = cSE(out_channels, act_fun=self.act_fn)
        

        self.conv1 = ConvBn2d(in_channels, out_channels, kernel_size=3, padding=1,norm_layer= norm_layer)
        # self.conv2 = ConvBn2d(out_channels, out_channels, kernel_size=3, padding=1, norm_layer=norm_layer)
        # self.conv_res = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
      
    def forward(self, x):
        # residual = self.conv_res(x)
        x = self.conv1(x)
        x = self.act_fn(x)
        # x = self.conv2(x)

        if self.scse:
            # x = self.se(x)
            g1 = self.spatial_gate(x)
            #print('g1',g1.size())
            g2 = self.channel_gate(x)
            #print('g2',g2.size())
            x = g1*x + g2*x

        # x += residual
        # x = self.act_fn(x)
        return x


class DLinkNet(nn.Module):
    def __init__(self, in_channels, n_classes, backbone, model_path, scse=False, use_mish=False,use_cc=True):
        super(DLinkNet, self).__init__()
        if backbone == 'resnet34':
            filters = [64, 128, 256, 512]
            base_model = resnet34(pretrained=False)
        elif backbone == 'resnet50':
            filters = [256, 512, 1024, 2048]
            base_model = resnet50(pretrained=False)
        elif backbone == 'resnet101':
            filters = [256, 512, 1024, 2048]
            base_model = resnet101(pretrained=False)
        elif backbone == 'resnest50':
            filters = [256, 512, 1024, 2048]
            base_model = resnest50(pretrained=False)
        elif backbone == 'resnest101':
            filters = [256, 512, 1024, 2048]
            base_model = resnest101(pretrained=False)
        base_model.load_param(model_path)

        # input_dim >3
        if in_channels > 3:
            with torch.no_grad():
                pretrained_conv1 = base_model.conv1.weight.clone()
                base_model.conv1 = torch.nn.Conv2d(in_channels, 64, 7, 2, 3, bias=False)
                torch.nn.init.kaiming_normal_(
                    base_model.conv1.weight, mode='fan_out', nonlinearity='relu')
                # Re-assign pretraiend weights to first 3 channels
                # (assuming alpha channel is last in your input data)
                base_model.conv1.weight[:, :3] = pretrained_conv1

        self.use_scse = scse
        self.use_mish = use_mish

  

        self.firstconv = base_model.conv1
        self.firstbn = base_model.bn1
        self.firstrelu = base_model.relu
        # self.firstmaxpool = base_model.maxpool
        self.encoder1 = base_model.layer1

        self.encoder2 = base_model.layer2

        self.encoder3 = base_model.layer3

        self.encoder4 = base_model.layer4
        
        self.dblock = Dblock(filters[-1])
        # self.center = CenterBlock(filters[3], filters[3], scse=scse)

        self.use_cc = use_cc
        if self.use_cc:
            self.cc_head = RCCAModule(filters[3], filters[3])
            self.cc_conv = nn.Conv2d(filters[3] + filters[3], filters[3], kernel_size=1, stride=1, bias=False)

        # if self.use_gem:
        #     self.maxpool = AdaptiveGeM2d(output_size=(8, 8), eps=1e-6, freeze_p=False)
        # else:
        #     self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        if self.use_mish:
            self.act_fn = Mish()
        else:
            self.act_fn = nonlinearity


        self.decoder4 = DecoderBlock(filters[3], filters[2], scse=False, act_fn=self.act_fn)
        self.decoder3 = DecoderBlock(filters[2], filters[1], scse=scse, act_fn=self.act_fn)
        self.decoder2 = DecoderBlock(filters[1], filters[0], scse=scse, act_fn=self.act_fn)
        self.decoder1 = DecoderBlock(filters[0], filters[0], scse=scse, act_fn=self.act_fn)

        # self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)

        # self.finalrelu1 = self.act_fn      

        self.finalconv2_0 = nn.Conv2d(filters[0], 64, 3, padding=1)

        self.finalconv2_1 = nn.Conv2d(filters[0], 64, 3, padding=1)
        self.finalconv2_2 = nn.Conv2d(filters[1], 64, 3, padding=1)
        self.finalconv2_3 = nn.Conv2d(filters[2], 64, 3, padding=1)
       
        self.finalrelu2 = self.act_fn
        # self.finalrelu2 = nonlinearity

        self.finalconv3 = nn.Conv2d(64, n_classes, 3, padding=1)

    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        # x = self.firstmaxpool(x)
        e1 = self.encoder1(x)

        e2 = self.encoder2(e1)

        e3 = self.encoder3(e2)

        e4 = self.encoder4(e3)
        
        # Center
        f = self.center(e4)

        if self.use_cc:
            cc_f = self.cc_head(f)
            f = self.cc_conv(torch.cat([e4, cc_f],1))

        # f = self.maxpool(f)

        # Decoder
        # print(f.shape,self.decoder4(f).shape, e4.shape)
        d4 = self.decoder4(f) + e3
        d3 = self.decoder3(d4) + e2 # 
        d2 = self.decoder2(d3) + e1 # 128
        d1 = self.decoder1(d2)
        
        # out = self.finaldeconv1(d1)
        # out = self.finalrelu1(out)

        out = self.finalconv2_0(d1)
        out = self.finalrelu2(out)

        out1 = self.finalconv2_1(d2)
        out1 = self.finalrelu2(out1)

        out2 = self.finalconv2_2(d3)
        out2 = self.finalrelu2(out2)

        out3 = self.finalconv2_3(d4)
        out3 = self.finalrelu2(out3)

        out = self.finalconv3(out)
        out1 = self.finalconv3(out1)
        out2 = self.finalconv3(out2)
        out3 = self.finalconv3(out3)

        outs = [
                out,
                F.upsample(out1, scale_factor=2,  mode='bilinear', align_corners=False), 
                F.upsample(out2, scale_factor=4,  mode='bilinear', align_corners=False),
                F.upsample(out3, scale_factor=8,  mode='bilinear', align_corners=False),
            ]

        out = torch.max(torch.stack(outs), dim=0)[0]
       
        return out