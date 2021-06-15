import torch
from torch import nn
import torch.nn.functional as F


from .modules.utils import initialize_weights
from .modules.backbones.resnet import resnet18, resnet34, resnet50, resnet101
from .modules.backbones.resnet_ibn import resnet18_ibn_a, resnet34_ibn_a, resnet50_ibn_a, resnet101_ibn_a

Norm2d = nn.BatchNorm2d


class ConvBnRelu(nn.Module):
    # https://github.com/lingtengqiu/Deeperlab-pytorch/blob/master/seg_opr/seg_oprs.py
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0,
                 norm_layer=Norm2d):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, bias=False)
        self.bn = norm_layer(out_planes, eps=1e-5)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


class AtrousSpatialPyramidPoolingModule(nn.Module):
    """
    operations performed:
      1x1 x depth
      3x3 x depth dilation 6
      3x3 x depth dilation 12
      3x3 x depth dilation 18
      image pooling
      concatenate all together
      Final 1x1 conv
    """

    def __init__(self, in_dim, reduction_dim=256, output_stride=16,
                 rates=(6, 12, 18)):
        super(AtrousSpatialPyramidPoolingModule, self).__init__()

        if output_stride == 8:
            rates = [2 * r for r in rates]
        elif output_stride == 16:
            pass
        else:
            raise 'output stride of {} not supported'.format(output_stride)

        self.features = []
        # 1x1
        self.features.append(
            nn.Sequential(nn.Conv2d(in_dim, reduction_dim, kernel_size=1,
                                    bias=False),
                          Norm2d(reduction_dim), nn.ReLU(inplace=True)))
        # other rates
        for r in rates:
            self.features.append(nn.Sequential(
                nn.Conv2d(in_dim, reduction_dim, kernel_size=3,
                          dilation=r, padding=r, bias=False),
                Norm2d(reduction_dim),
                nn.ReLU(inplace=True)
            ))
        self.features = nn.ModuleList(self.features)

        # img level features
        self.img_pooling = nn.AdaptiveAvgPool2d(1)
        self.img_conv = nn.Sequential(
            nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
            Norm2d(reduction_dim), nn.ReLU(inplace=True))

    def forward(self, x):
        x_size = x.size()

        img_features = self.img_pooling(x)
        img_features = self.img_conv(img_features)
        img_features = F.upsample(img_features, x_size[2:], mode='bilinear', align_corners=False)
        out = img_features

        for f in self.features:
            y = f(x)
            out = torch.cat((out, y), 1)
        return out


class ASPP_edge(AtrousSpatialPyramidPoolingModule):
    def __init__(self, in_dim, reduction_dim=256, output_stride=16,
                 rates=(6, 12, 18)):
        super(ASPP_edge, self).__init__(in_dim=in_dim,
                                        reduction_dim=reduction_dim,
                                        output_stride=output_stride,
                                        rates=rates)
        self.edge_conv = nn.Sequential(
            nn.Conv2d(1, reduction_dim, kernel_size=1, bias=False),
            Norm2d(reduction_dim), nn.ReLU(inplace=True))

    def forward(self, x, edge):
        x_size = x.size()

        img_features = self.img_pooling(x)
        img_features = self.img_conv(img_features)
        img_features = F.upsample(img_features, x_size[2:], mode='bilinear', align_corners=False)
        out = img_features
        edge_features = F.upsample(edge, x_size[2:], mode='bilinear', align_corners=False)
        edge_features = self.edge_conv(edge_features)
        out = torch.cat((out, edge_features), 1)

        for f in self.features:
            y = f(x)
            out = torch.cat((out, y), 1)
        return out


def dpc_conv(in_dim, reduction_dim, dil, separable):
    if separable:
        groups = reduction_dim
    else:
        groups = 1

    return nn.Sequential(
        nn.Conv2d(in_dim, reduction_dim, kernel_size=3, dilation=dil,
                  padding=dil, bias=False, groups=groups),
        nn.BatchNorm2d(reduction_dim),
        nn.ReLU(inplace=True)
    )


class DPC(nn.Module):
    '''
    From: Searching for Efficient Multi-scale architectures for dense
    prediction
    '''
    def __init__(self, in_dim, reduction_dim=256, output_stride=16,
                 rates=[(1, 6), (18, 15), (6, 21), (1, 1), (6, 3)],
                 dropout=False, separable=False):
        super(DPC, self).__init__()

        self.dropout = dropout
        if output_stride == 8:
            rates = [(2 * r[0], 2 * r[1]) for r in rates]
        elif output_stride == 16:
            pass
        else:
            raise 'output stride of {} not supported'.format(output_stride)

        self.a = dpc_conv(in_dim, reduction_dim, rates[0], separable)
        self.b = dpc_conv(reduction_dim, reduction_dim, rates[1], separable)
        self.c = dpc_conv(reduction_dim, reduction_dim, rates[2], separable)
        self.d = dpc_conv(reduction_dim, reduction_dim, rates[3], separable)
        self.e = dpc_conv(reduction_dim, reduction_dim, rates[4], separable)

        self.drop = nn.Dropout(p=0.1)

    def forward(self, x):
        a = self.a(x)
        b = self.b(a)
        c = self.c(a)
        d = self.d(a)
        e = self.e(b)
        out = torch.cat((a, b, c, d, e), 1)
        if self.dropout:
            out = self.drop(out)
        return out


def get_aspp(high_level_ch, bottleneck_ch, output_stride, dpc=False):
    """
    Create aspp block
    """
    if dpc:
        aspp = DPC(high_level_ch, bottleneck_ch, output_stride=output_stride)
    else:
        aspp = AtrousSpatialPyramidPoolingModule(high_level_ch, bottleneck_ch,
                                                 output_stride=output_stride)
    aspp_out_ch = 5 * bottleneck_ch
    return aspp, aspp_out_ch


class DeepV3Plus(nn.Module):
    """
    DeepLabV3+ with various trunks supported
    Always stride8
    """
    def __init__(self, in_channels, n_classes, backbone, model_path, use_dpc=False, init_all=False):
        super(DeepV3Plus, self).__init__()

        if backbone == 'resnet50':
            self.base_model = resnet50(pretrained=False)
            s2_ch = 256
            high_level_ch = 2048
        if backbone == 'resnet101':
            self.base_model = resnet101(pretrained=False)
            s2_ch = 256
            high_level_ch = 2048

        if backbone == 'resnet50_ibn_a':
            self.base_model = resnet50_ibn_a(pretrained=False)
            s2_ch = 256
            high_level_ch = 2048

        if backbone == 'resnet101_ibn_a':
            self.base_model = resnet101_ibn_a(pretrained=False)
            s2_ch = 256
            high_level_ch = 2048

        if model_path:
            self.base_model.load_param(model_path)

        # input_dim >3
        if in_channels > 3:
            with torch.no_grad():
                pretrained_conv1 = self.base_model.conv1.weight.clone()
                self.base_model.conv1 = torch.nn.Conv2d(in_channels, 64, 7, 2, 3, bias=False)
                torch.nn.init.kaiming_normal_(
                    self.base_model.conv1.weight, mode='fan_out', nonlinearity='relu')
                # Re-assign pretraiend weights to first 3 channels
                # (assuming alpha channel is last in your input data)
                self.base_model.conv1.weight[:, :3] = pretrained_conv1

        self.aspp, aspp_out_ch = get_aspp(high_level_ch,
                                          bottleneck_ch=256,
                                          output_stride=8,
                                          dpc=use_dpc)
        self.bot_fine = nn.Conv2d(s2_ch, 48, kernel_size=1, bias=False)
        self.bot_aspp = nn.Conv2d(aspp_out_ch, 256, kernel_size=1, bias=False)
        self.final = nn.Sequential(
            nn.Conv2d(256 + 48, 256, kernel_size=3, padding=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, n_classes, kernel_size=1, bias=False))

        if init_all:
            initialize_weights(self.aspp)
            initialize_weights(self.bot_aspp)
            initialize_weights(self.bot_fine)
            initialize_weights(self.final)
        else:
            initialize_weights(self.final)

    def forward(self, x):
        x_size = x.size()
        s2_features, _, final_features = self.base_model.forward_features(x)
        aspp = self.aspp(final_features)
        conv_aspp = self.bot_aspp(aspp)
        conv_s2 = self.bot_fine(s2_features)
        conv_aspp = F.upsample(conv_aspp, s2_features.size()[2:],mode='bilinear', align_corners=False)
        cat_s4 = [conv_s2, conv_aspp]
        cat_s4 = torch.cat(cat_s4, 1)
        final = self.final(cat_s4)
        out = F.upsample(final, x_size[2:], mode='bilinear', align_corners=False)

        return out