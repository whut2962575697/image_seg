import torch.nn as nn
import torch
import torchvision
import torch.nn.functional as F
from functools import partial
from .modules.scse import sSE, cSE
from .modules.backbones.resnet import resnet18, resnet34, resnet50, resnet101

from .modules.backbones.resnest import resnest50, resnest101, resnest200
from .modules.backbones.resnet_ibn import resnet34_ibn_a, resnet18_ibn_a, resnet34_ibn_b, resnet18_ibn_b,\
    resnet101_ibn_a, resnet101_ibn_b, resnet50_ibn_a
from .modules.backbones.resnext_ibn import resnext101_ibn_a
from .modules.backbones.se_ibn import se_resnet101_ibn_a

from .modules.backbones.res2net_dal import res2net_dla60, res2next_dla60

from .modules.backbones.sw_resnet import resnet101 as sw_resnet101

from .modules.dblock import Dblock

from .modules.mish import Mish

from .modules.cc import CC_module as CrissCrossAttention

from .modules.base_ocr_model import BaseOC_Module
from .modules.diversebranchblock import DiverseBranchBlock

nonlinearity = partial(F.relu, inplace=True)


class RCCAModule(nn.Module):
    def __init__(self, in_channels, out_channels, act_fn=nn.ReLU(inplace=True)):
        super(RCCAModule, self).__init__()
        inter_channels = in_channels // 4
        self.conva = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(inter_channels), act_fn)
        self.cca = CrissCrossAttention(inter_channels)
        self.convb = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(inter_channels), act_fn)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels + inter_channels, out_channels, kernel_size=3, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(out_channels), act_fn,
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


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1,
                 act_fn=nn.ReLU(inplace=True)):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = act_fn

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class Decoder(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, scse=False, act_fn=nn.ReLU(inplace=True)):
        super(Decoder, self).__init__()

        self.act_fn = act_fn
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

        self.conv_bn_relu = DiverseBranchBlock(in_channels=middle_channels, out_channels=out_channels, kernel_size=3,
                                               stride=1, padding=1,
                                               deploy=False, nonlinear=self.act_fn)
        self.scse = scse

        if self.scse:
            self.spatial_gate = sSE(out_channels)
            self.channel_gate = cSE(out_channels, self.act_fn)

    def forward(self, x1, x2, direct_cat=False):
        if not direct_cat:
            x1 = self.up(x1)
            # x1 = self.act_fn(x1)

        x1 = torch.cat((x1, x2), dim=1)
        ### todo
        x1 = self.conv_bn_relu(x1)
        if self.scse:
            g1 = self.spatial_gate(x1)
            # print('g1',g1.size())
            g2 = self.channel_gate(x1)
            # print('g2',g2.size())
            x1 = g1 * x1 + g2 * x1  # skip connect?
        return x1


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
    def __init__(self, in_channels, out_channels, scse=False, norm_layer=nn.BatchNorm2d, act_fn=nn.ReLU(inplace=True)):
        super(CenterBlock, self).__init__()
        self.scse = scse

        self.act_fn = act_fn
        if self.scse:
            self.spatial_gate = sSE(out_channels)
            self.channel_gate = cSE(out_channels, act_fun=self.act_fn)

        self.conv1 = DiverseBranchBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1,
                                        padding=1,
                                        deploy=False)
        self.conv2 = DiverseBranchBlock(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1,
                                        padding=1,
                                        deploy=False)
        self.conv_res = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        residual = self.conv_res(x)
        x = self.conv1(x)
        x = self.act_fn(x)
        x = self.conv2(x)

        if self.scse:
            # x = self.se(x)
            g1 = self.spatial_gate(x)
            # print('g1',g1.size())
            g2 = self.channel_gate(x)
            # print('g2',g2.size())
            x = g1 * x + g2 * x

        x += residual
        x = self.act_fn(x)
        return x


class SEB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SEB, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear")

    def forward(self, x, up_scale=True):
        x1, x2 = x
        # print(x1.shape, x2.shape)
        if up_scale:
            return x1 * self.upsample(self.conv(x2))
        else:
            return x1 * self.conv(x2)


class Res_Unet(nn.Module):
    def __init__(self, in_channels, n_classes, backbone, model_path, dropout_rate=0.0, scse=False, use_mish=False,
                 db_block=False, use_spp=False, hypercolumn=False, use_cc=False, deepsupversion=False, decode_dim=128):
        super().__init__()
        self.deepsupversion = deepsupversion

        self.hypercolumn = hypercolumn

        if use_mish:
            self.act_fn = Mish()
        else:
            self.act_fn = nn.ReLU(inplace=True)
            # self.act_fn = nn.LeakyReLU(inplace=True)
        if backbone == 'resnet18':
            filters = [64, 128, 256, 512]
            base_model = resnet18(pretrained=False)
            if decode_dim is None:
                self.decode_dim = 64
            else:
                self.decode_dim = decode_dim
        elif backbone == 'resnet34':
            filters = [64, 128, 256, 512]
            base_model = resnet34(pretrained=False)
            if decode_dim is None:
                self.decode_dim = 64
            else:
                self.decode_dim = decode_dim

        elif backbone == 'resnet34_ibn_a':
            filters = [64, 128, 256, 512]
            base_model = resnet34_ibn_a(pretrained=False)
            if decode_dim is None:
                self.decode_dim = 64
            else:
                self.decode_dim = decode_dim
            filter0 = 64
        elif backbone == 'resnet101_ibn_a':
            filters = [256, 512, 1024, 2048]
            base_model = resnet101_ibn_a(pretrained=False)
            if decode_dim is None:
                self.decode_dim = 256
            else:
                self.decode_dim = decode_dim
            filter0 = 64
        elif backbone == 'resnet50_ibn_a':
            filters = [256, 512, 1024, 2048]
            base_model = resnet50_ibn_a(pretrained=False)
            if decode_dim is None:
                self.decode_dim = 256
            else:
                self.decode_dim = decode_dim
            filter0 = 64
        elif backbone == 'resnext101_ibn_a':
            filters = [256, 512, 1024, 2048]
            base_model = resnext101_ibn_a(pretrained=False)
            if decode_dim is None:
                self.decode_dim = 256
            else:
                self.decode_dim = decode_dim
            filter0 = 64
        elif backbone == 'se_resnet101_ibn_a':
            filters = [256, 512, 1024, 2048]
            base_model = se_resnet101_ibn_a(pretrained=False)
            if decode_dim is None:
                self.decode_dim = 256
            else:
                self.decode_dim = decode_dim
            filter0 = 64
        elif backbone == "sw_resnet101":
            filters = [256, 512, 1024, 2048]
            base_model = sw_resnet101(pretrained=False)
            if decode_dim is None:
                self.decode_dim = 256
            else:
                self.decode_dim = decode_dim
        elif backbone == 'resnet101_ibn_b':
            filters = [256, 512, 1024, 2048]
            base_model = resnet101_ibn_b(pretrained=False)
            if decode_dim is None:
                self.decode_dim = 256
            else:
                self.decode_dim = decode_dim
            filter0 = 64
        elif backbone == 'resnet18_ibn_a':
            filters = [64, 128, 256, 512]
            base_model = resnet18_ibn_a(pretrained=False)
            if decode_dim is None:
                self.decode_dim = 64
            else:
                self.decode_dim = decode_dim
            filter0 = 64
        elif backbone == 'resnet34_ibn_b':
            filters = [64, 128, 256, 512]
            base_model = resnet34_ibn_b(pretrained=False)
            if decode_dim is None:
                self.decode_dim = 64
            else:
                self.decode_dim = decode_dim

        elif backbone == 'resnet18_ibn_b':
            filters = [64, 128, 256, 512]
            base_model = resnet18_ibn_b(pretrained=False)
            if decode_dim is None:
                self.decode_dim = 64
            else:
                self.decode_dim = decode_dim
        elif backbone == 'resnet50':
            filters = [256, 512, 1024, 2048]
            base_model = resnet50(pretrained=False)
            if decode_dim is None:
                self.decode_dim = 256
            else:
                self.decode_dim = decode_dim
        elif backbone == 'resnet101':
            filters = [256, 512, 1024, 2048]
            base_model = resnet101(pretrained=False)
            if decode_dim is None:
                self.decode_dim = 256
            else:
                self.decode_dim = decode_dim
            filter0 = 64
        elif backbone == 'resnest50':
            filters = [256, 512, 1024, 2048]
            base_model = resnest50(pretrained=False)
            if decode_dim is None:
                self.decode_dim = 256
            else:
                self.decode_dim = decode_dim
        elif backbone == 'resnest101':
            filters = [256, 512, 1024, 2048]
            base_model = resnest101(pretrained=False)
            if decode_dim is None:
                self.decode_dim = 256
            else:
                self.decode_dim = decode_dim
            filter0 = 128
        elif backbone == 'res2next_dla60':
            filters = [128, 256, 512, 1024]
            base_model = res2next_dla60(pretrained=False)
            if decode_dim is None:
                self.decode_dim = 128
            else:
                self.decode_dim = decode_dim
        if model_path:
            base_model.load_param(model_path)

        # input_dim >3
        if in_channels > 3:
            with torch.no_grad():
                pretrained_conv1 = base_model.conv1.weight.clone()
                base_model.conv1 = torch.nn.Conv2d(4, 64, 7, 2, 3, bias=False)
                torch.nn.init.kaiming_normal_(
                    base_model.conv1.weight, mode='fan_out', nonlinearity='relu')
                # Re-assign pretraiend weights to first 3 channels
                # (assuming alpha channel is last in your input data)
                base_model.conv1.weight[:, :3] = pretrained_conv1

        self.use_scse = scse
        norm_layer = nn.BatchNorm2d

        self.base_layer = nn.Sequential(
            base_model.conv1,
            base_model.bn1,
            base_model.relu,
        )

        # self.firstmaxpool = base_model.maxpool
        self.encoder1 = base_model.layer1

        self.encoder2 = base_model.layer2

        self.encoder3 = base_model.layer3

        self.encoder4 = base_model.layer4

        self.use_dblock = db_block

        if db_block:
            self.center = Dblock(filters[3], self.decode_dim, self.act_fn)

        else:
            self.center = CenterBlock(filters[3], self.decode_dim, scse=scse, norm_layer=norm_layer, act_fn=self.act_fn)

        self.use_dropout = False

        self.use_cc = use_cc
        if self.use_cc:
            self.cc_head = RCCAModule(self.decode_dim, self.decode_dim, act_fn=self.act_fn)
            self.cc_conv = nn.Conv2d(filters[3] + self.decode_dim, self.decode_dim, kernel_size=1, stride=1, bias=False)

        self.seb2 = SEB(self.decode_dim, filters[1])
        self.seb3 = SEB(self.decode_dim, filters[0])
        self.seb4 = SEB(self.decode_dim, filter0)

        self.use_spp = use_spp

        self.decode4 = Decoder(self.decode_dim, filters[2] + self.decode_dim, self.decode_dim, scse=scse,
                               act_fn=self.act_fn)
        self.decode3 = Decoder(self.decode_dim, filters[1] + self.decode_dim, self.decode_dim, scse=scse,
                               act_fn=self.act_fn)
        self.decode2 = Decoder(self.decode_dim, filters[0] + self.decode_dim, self.decode_dim, scse=scse,
                               act_fn=self.act_fn)

        self.decode1 = Decoder(self.decode_dim, filter0 + self.decode_dim, self.decode_dim, scse=scse,
                               act_fn=self.act_fn)
        self.decode0 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),

            DiverseBranchBlock(in_channels=self.decode_dim, out_channels=self.decode_dim // 2, kernel_size=3,
                               stride=1, padding=1,
                               deploy=False, nonlinear=self.act_fn),


            DiverseBranchBlock(in_channels=self.decode_dim // 2, out_channels=self.decode_dim, kernel_size=3,
                               stride=1, padding=1,
                               deploy=False, nonlinear=self.act_fn),

        )

        if dropout_rate > 0:
            self.use_dropout = True
            self.dropout = nn.Dropout2d(dropout_rate)
        if self.hypercolumn:
            self.fuse_conv = nn.Sequential(
                nn.Conv2d(self.decode_dim * 4, self.decode_dim, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(self.decode_dim),
                self.act_fn,
                nn.Dropout2d(0.1),
                )

        self.conv_last = nn.Conv2d(self.decode_dim, n_classes, 1)


    def freeze_bn(self):
        print("freeze bacth normalization successfully!")
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.InstanceNorm2d):
                m.eval()

    def forward(self, input):
        x = self.base_layer(input)
        # x = self.firstmaxpool(x)
        e1 = self.encoder1(x)

        e2 = self.encoder2(e1)

        e3 = self.encoder3(e2)

        e4 = self.encoder4(e3)

        f = self.center(e4)

        if self.use_cc:
            cc_f = self.cc_head(f)
            f = self.cc_conv(torch.cat([e4, cc_f], 1))

        d4 = self.decode4(f, e3)  # 256,16,16

        eb2 = self.seb2([e2, F.upsample(f, scale_factor=4, mode='bilinear', align_corners=True)], up_scale=False)
        d3 = self.decode3(d4, eb2)  # 256,32,32

        eb1 = self.seb3([e1, F.upsample(f, scale_factor=8, mode='bilinear', align_corners=True)], up_scale=False)
        d2 = self.decode2(d3, eb1)  # 128,64,64

        eb0 = self.seb4([x, F.upsample(f, scale_factor=8, mode='bilinear', align_corners=True)], up_scale=False)
        d1 = self.decode1(d2, eb0, direct_cat=True)  # 64,128,128
        d0 = self.decode0(d1)


        out = d1
        if self.deepsupversion:
            outs = [
                out,
                F.upsample(self.conv_last(d2), scale_factor=2, mode='bilinear', align_corners=False),
                F.upsample(self.conv_last(d3), scale_factor=4, mode='bilinear', align_corners=False),
                F.upsample(self.conv_last(d4), scale_factor=8, mode='bilinear', align_corners=False),
            ]
            outs = [torch.max(torch.stack(outs), dim=0)[0]] + outs
            return outs
        else:
            outs = [
                out,
                F.upsample(self.conv_last(d1), scale_factor=2, mode='bilinear', align_corners=True),
                F.upsample(self.conv_last(d2), scale_factor=2, mode='bilinear', align_corners=True),
                F.upsample(self.conv_last(d3), scale_factor=4, mode='bilinear', align_corners=True),
                F.upsample(self.conv_last(d4), scale_factor=8, mode='bilinear', align_corners=True),

            ]

            out = torch.max(torch.stack(outs), dim=0)[0]
            return out

    def load_param(self, model_path):
        param_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
        if 'state_dict' in param_dict.keys():
            param_dict = param_dict['state_dict']

        start_with_module = False
        for k in param_dict.keys():
            if k.startswith('module.'):
                start_with_module = True
                break
        if start_with_module:
            param_dict = {k[7:]: v for k, v in param_dict.items()}

        print('ignore_param:')
        print(
            [k for k, v in param_dict.items() if k not in self.state_dict() or self.state_dict()[k].size() != v.size()])
        print('unload_param:')
        print(
            [k for k, v in self.state_dict().items() if k not in param_dict.keys() or param_dict[k].size() != v.size()])

        param_dict = {k: v for k, v in param_dict.items() if
                      k in self.state_dict() and self.state_dict()[k].size() == v.size()}
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])