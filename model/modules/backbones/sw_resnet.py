import math
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

# from ..utils import build_norm_layer


__all__ = ['resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}





import torch.nn as nn
# from models.ops import SwitchWhiten2d





import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class SwitchWhiten2d(Module):
    """Switchable Whitening.
    Args:
        num_features (int): Number of channels.
        num_pergroup (int): Number of channels for each whitening group.
        sw_type (int): Switchable whitening type, from {2, 3, 5}.
            sw_type = 2: BW + IW
            sw_type = 3: BW + IW + LN
            sw_type = 5: BW + IW + BN + IN + LN
        T (int): Number of iterations for iterative whitening.
        tie_weight (bool): Use the same importance weight for mean and
            covariance or not.
    """

    def __init__(self,
                 num_features,
                 num_pergroup=16,
                 sw_type=2,
                 T=5,
                 tie_weight=False,
                 eps=1e-5,
                 momentum=0.99,
                 affine=True):
        super(SwitchWhiten2d, self).__init__()
        if sw_type not in [2, 3, 5]:
            raise ValueError('sw_type should be in [2, 3, 5], '
                             'but got {}'.format(sw_type))
        assert num_features % num_pergroup == 0
        self.num_features = num_features
        self.num_pergroup = num_pergroup
        self.num_groups = num_features // num_pergroup
        self.sw_type = sw_type
        self.T = T
        self.tie_weight = tie_weight
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        num_components = sw_type

        self.sw_mean_weight = Parameter(torch.ones(num_components))
        if not self.tie_weight:
            self.sw_var_weight = Parameter(torch.ones(num_components))
        else:
            self.register_parameter('sw_var_weight', None)

        if self.affine:
            self.weight = Parameter(torch.ones(num_features))
            self.bias = Parameter(torch.zeros(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.register_buffer('running_mean',
                             torch.zeros(self.num_groups, num_pergroup, 1))
        self.register_buffer(
            'running_cov',
            torch.eye(num_pergroup).unsqueeze(0).repeat(self.num_groups, 1, 1))

        self.reset_parameters()

    def reset_parameters(self):
        self.running_mean.zero_()
        self.running_cov.zero_()
        nn.init.ones_(self.sw_mean_weight)
        if not self.tie_weight:
            nn.init.ones_(self.sw_var_weight)
        if self.affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def __repr__(self):
        return ('{name}({num_features}, num_pergroup={num_pergroup}, '
                'sw_type={sw_type}, T={T}, tie_weight={tie_weight}, '
                'eps={eps}, momentum={momentum}, affine={affine})'.format(
                    name=self.__class__.__name__, **self.__dict__))

    def forward(self, x):
        N, C, H, W = x.size()
        c, g = self.num_pergroup, self.num_groups

        in_data_t = x.transpose(0, 1).contiguous()
        # g x c x (N x H x W)
        in_data_t = in_data_t.view(g, c, -1)

        # calculate batch mean and covariance
        if self.training:
            # g x c x 1
            mean_bn = in_data_t.mean(-1, keepdim=True)
            in_data_bn = in_data_t - mean_bn
            # g x c x c
            cov_bn = torch.bmm(in_data_bn,
                               in_data_bn.transpose(1, 2)).div(H * W * N)

            self.running_mean.mul_(self.momentum)
            self.running_mean.add_((1 - self.momentum) * mean_bn.data)
            self.running_cov.mul_(self.momentum)
            self.running_cov.add_((1 - self.momentum) * cov_bn.data)
        else:
            mean_bn = torch.autograd.Variable(self.running_mean)
            cov_bn = torch.autograd.Variable(self.running_cov)

        mean_bn = mean_bn.view(1, g, c, 1).expand(N, g, c, 1).contiguous()
        mean_bn = mean_bn.view(N * g, c, 1)
        cov_bn = cov_bn.view(1, g, c, c).expand(N, g, c, c).contiguous()
        cov_bn = cov_bn.view(N * g, c, c)

        # (N x g) x c x (H x W)
        in_data = x.view(N * g, c, -1)

        eye = in_data.data.new().resize_(c, c)
        eye = torch.nn.init.eye_(eye).view(1, c, c).expand(N * g, c, c)

        # calculate other statistics
        # (N x g) x c x 1
        mean_in = in_data.mean(-1, keepdim=True)
        x_in = in_data - mean_in
        # (N x g) x c x c
        cov_in = torch.bmm(x_in, torch.transpose(x_in, 1, 2)).div(H * W)
        if self.sw_type in [3, 5]:
            x = x.view(N, -1)
            mean_ln = x.mean(-1, keepdim=True).view(N, 1, 1, 1)
            mean_ln = mean_ln.expand(N, g, 1, 1).contiguous().view(N * g, 1, 1)
            var_ln = x.var(-1, keepdim=True).view(N, 1, 1, 1)
            var_ln = var_ln.expand(N, g, 1, 1).contiguous().view(N * g, 1, 1)
            var_ln = var_ln * eye
        if self.sw_type == 5:
            var_bn = torch.diag_embed(torch.diagonal(cov_bn, dim1=-2, dim2=-1))
            var_in = torch.diag_embed(torch.diagonal(cov_in, dim1=-2, dim2=-1))

        # calculate weighted average of mean and covariance
        softmax = nn.Softmax(0)
        mean_weight = softmax(self.sw_mean_weight)
        if not self.tie_weight:
            var_weight = softmax(self.sw_var_weight)
        else:
            var_weight = mean_weight

        # BW + IW
        if self.sw_type == 2:
            # (N x g) x c x 1
            mean = mean_weight[0] * mean_bn + mean_weight[1] * mean_in
            cov = var_weight[0] * cov_bn + var_weight[1] * cov_in + \
                self.eps * eye
        # BW + IW + LN
        elif self.sw_type == 3:
            mean = mean_weight[0] * mean_bn + \
                mean_weight[1] * mean_in + mean_weight[2] * mean_ln
            cov = var_weight[0] * cov_bn + var_weight[1] * cov_in + \
                var_weight[2] * var_ln + self.eps * eye
        # BW + IW + BN + IN + LN
        elif self.sw_type == 5:
            mean = (mean_weight[0] + mean_weight[2]) * mean_bn + \
                (mean_weight[1] + mean_weight[3]) * mean_in + \
                mean_weight[4] * mean_ln
            cov = var_weight[0] * cov_bn + var_weight[1] * cov_in + \
                var_weight[0] * var_bn + var_weight[1] * var_in + \
                var_weight[4] * var_ln + self.eps * eye

        # perform whitening using Newton's iteration
        Ng, c, _ = cov.size()
        P = torch.eye(c).to(cov).expand(Ng, c, c)
        # reciprocal of trace of covariance
        rTr = (cov * P).sum((1, 2), keepdim=True).reciprocal_()
        cov_N = cov * rTr
        for k in range(self.T):
            P = torch.baddbmm(1.5, P, -0.5, torch.matrix_power(P, 3), cov_N)
        # whiten matrix: the matrix inverse of covariance, i.e., cov^{-1/2}
        wm = P.mul_(rTr.sqrt())

        x_hat = torch.bmm(wm, in_data - mean)
        x_hat = x_hat.view(N, C, H, W)
        if self.affine:
            x_hat = x_hat * self.weight.view(1, self.num_features, 1, 1) + \
                self.bias.view(1, self.num_features, 1, 1)

        return x_hat



norm_cfg = {
    # format: layer_type: (abbreviation, module)
    'BN': ('bn', nn.BatchNorm2d),
    'GN': ('gn', nn.GroupNorm),
    'SW': ('sw', SwitchWhiten2d)
}


def build_norm_layer(cfg, num_features, postfix=''):
    """ Build normalization layer
    Args:
        cfg (dict): cfg should contain:
            type (str): identify norm layer type.
            layer args: args needed to instantiate a norm layer.
            requires_grad (bool): [optional] whether stop gradient updates
        num_features (int): number of channels from input.
        postfix (int, str): appended into norm abbreviation to
            create named layer.
    Returns:
        name (str): abbreviation + postfix
        layer (nn.Module): created norm layer
    """
    assert isinstance(cfg, dict) and 'type' in cfg
    cfg_ = cfg.copy()

    layer_type = cfg_.pop('type')
    if layer_type not in norm_cfg:
        raise KeyError('Unrecognized norm type {}'.format(layer_type))
    else:
        abbr, norm_layer = norm_cfg[layer_type]
        if norm_layer is None:
            raise NotImplementedError

    assert isinstance(postfix, (int, str))
    name = abbr + str(postfix)

    requires_grad = cfg_.pop('requires_grad', True)
    cfg_.setdefault('eps', 1e-5)
    if layer_type != 'GN':
        layer = norm_layer(num_features, **cfg_)
    else:
        assert 'num_groups' in cfg_
        layer = norm_layer(num_channels=num_features, **cfg_)

    for param in layer.parameters():
        param.requires_grad = requires_grad

    return name, layer



def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 norm_cfg=dict(type='BN'),
                 sw_cfg=None):
        super(BasicBlock, self).__init__()
        self.norm1_name, norm1 = build_norm_layer(
            sw_cfg if sw_cfg is not None else norm_cfg, planes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, planes, postfix=2)

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.add_module(self.norm1_name, norm1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.add_module(self.norm2_name, norm2)
        self.downsample = downsample
        self.stride = stride

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 norm_cfg=dict(type='BN'),
                 sw_cfg=None):
        super(Bottleneck, self).__init__()
        self.norm1_name, norm1 = build_norm_layer(norm_cfg, planes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(
            sw_cfg if sw_cfg is not None else norm_cfg, planes, postfix=2)
        self.norm3_name, norm3 = build_norm_layer(
            norm_cfg, planes * 4, postfix=3)

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.add_module(self.norm1_name, norm1)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.add_module(self.norm2_name, norm2)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.add_module(self.norm3_name, norm3)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    @property
    def norm3(self):
        return getattr(self, self.norm3_name)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.norm3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 num_classes=1000,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 sw_cfg=None,
                 stage_with_sw=(True, True, True, False)):
        self.inplanes = 64
        super(ResNet, self).__init__()

        self.norm_cfg = norm_cfg
        self.sw_cfg = sw_cfg
        self.stage_with_sw = stage_with_sw
        self.norm1_name, norm1 = build_norm_layer(
            sw_cfg if sw_cfg is not None else norm_cfg, 64, postfix=1)

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.add_module(self.norm1_name, norm1)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1,
                                       with_sw=stage_with_sw[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       with_sw=stage_with_sw[1])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       with_sw=stage_with_sw[2])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       with_sw=stage_with_sw[3])
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    def _make_layer(self, block, planes, blocks, stride, with_sw):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                build_norm_layer(self.norm_cfg, planes * block.expansion)[1],
            )

        layers = []
        layers.append(
            block(self.inplanes,
                  planes,
                  stride,
                  downsample,
                  norm_cfg=self.norm_cfg,
                  sw_cfg=None))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(self.inplanes,
                      planes,
                      norm_cfg=self.norm_cfg,
                      sw_cfg=self.sw_cfg if
                      (with_sw and i % 2 == 1) else None))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

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
            param_dict = {k[7:] : v for k, v in param_dict.items() }
  
        print('ignore_param:')
        print([k for k, v in param_dict.items() if k not in self.state_dict() or self.state_dict()[k].size() != v.size()])
        print('unload_param:')
        print([k for k, v in self.state_dict().items() if k not in param_dict.keys() or param_dict[k].size() != v.size()] )

        param_dict = {k: v for k, v in param_dict.items() if k in self.state_dict() and self.state_dict()[k].size() == v.size()}
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']),
                              strict=False)
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']),
                              strict=False)
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']),
                              strict=False)
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """

    sw_cfg = dict(type='SW',
                  sw_type=2,
                  num_pergroup=16,
                  T=5,
                  tie_weight=False,
                  momentum=0.9,
                  affine=True)

    model = ResNet(Bottleneck, [3, 4, 23, 3], sw_cfg=sw_cfg, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']),
                              strict=False)
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']),
                              strict=False)
    return model