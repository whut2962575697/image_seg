# """ 
# Ported to pytorch thanks to [tstandley](https://github.com/tstandley/Xception-PyTorch)
# @author: tstandley
# Adapted by cadene
# Creates an Xception Model as defined in:
# Francois Chollet
# Xception: Deep Learning with Depthwise Separable Convolutions
# https://arxiv.org/pdf/1610.02357.pdf
# This weights ported from the Keras implementation. Achieves the following performance on the validation set:
# Loss:0.9173 Prec@1:78.892 Prec@5:94.292
# REMEMBER to set your image size to 3x299x299 for both test and validation
# normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
#                                   std=[0.5, 0.5, 0.5])
# The resize parameter of the validation transform should be 333, and make sure to center crop at 299x299
# """
# import math
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.utils.model_zoo as model_zoo
# from torch.nn import init
# from net.sync_batchnorm import SynchronizedBatchNorm2d

# bn_mom = 0.0003
# __all__ = ['xception']

# model_urls = {
#     'xception': '/home/wangyude/.torch/models/xception_pytorch_imagenet.pth'#'http://data.lip6.fr/cadene/pretrainedmodels/xception-b5690688.pth'
# }

# class SeparableConv2d(nn.Module):
#     def __init__(self,in_channels,out_channels,kernel_size=1,stride=1,padding=0,dilation=1,bias=False,activate_first=True,inplace=True):
#         super(SeparableConv2d,self).__init__()
#         self.relu0 = nn.ReLU(inplace=inplace)
#         self.depthwise = nn.Conv2d(in_channels,in_channels,kernel_size,stride,padding,dilation,groups=in_channels,bias=bias)
#         self.bn1 = SynchronizedBatchNorm2d(in_channels, momentum=bn_mom)
#         self.relu1 = nn.ReLU(inplace=True)
#         self.pointwise = nn.Conv2d(in_channels,out_channels,1,1,0,1,1,bias=bias)
#         self.bn2 = SynchronizedBatchNorm2d(out_channels, momentum=bn_mom)
#         self.relu2 = nn.ReLU(inplace=True)
#         self.activate_first = activate_first
#     def forward(self,x):
#         if self.activate_first:
#             x = self.relu0(x)
#         x = self.depthwise(x)
#         x = self.bn1(x)
#         if not self.activate_first:
#             x = self.relu1(x)
#         x = self.pointwise(x)
#         x = self.bn2(x)
#         if not self.activate_first:
#             x = self.relu2(x)
#         return x


# class Block(nn.Module):
#     def __init__(self,in_filters,out_filters,strides=1,atrous=None,grow_first=True,activate_first=True,inplace=True):
#         super(Block, self).__init__()
#         if atrous == None:
#             atrous = [1]*3
#         elif isinstance(atrous, int):
#             atrous_list = [atrous]*3
#             atrous = atrous_list
#         idx = 0
#         self.head_relu = True
#         if out_filters != in_filters or strides!=1:
#             self.skip = nn.Conv2d(in_filters,out_filters,1,stride=strides, bias=False)
#             self.skipbn = SynchronizedBatchNorm2d(out_filters, momentum=bn_mom)
#             self.head_relu = False
#         else:
#             self.skip=None
        
#         self.hook_layer = None
#         if grow_first:
#             filters = out_filters
#         else:
#             filters = in_filters
#         self.sepconv1 = SeparableConv2d(in_filters,filters,3,stride=1,padding=1*atrous[0],dilation=atrous[0],bias=False,activate_first=activate_first,inplace=self.head_relu)
#         self.sepconv2 = SeparableConv2d(filters,out_filters,3,stride=1,padding=1*atrous[1],dilation=atrous[1],bias=False,activate_first=activate_first)
#         self.sepconv3 = SeparableConv2d(out_filters,out_filters,3,stride=strides,padding=1*atrous[2],dilation=atrous[2],bias=False,activate_first=activate_first,inplace=inplace)

#     def forward(self,inp):
        
#         if self.skip is not None:
#             skip = self.skip(inp)
#             skip = self.skipbn(skip)
#         else:
#             skip = inp

#         x = self.sepconv1(inp)
#         x = self.sepconv2(x)
#         self.hook_layer = x
#         x = self.sepconv3(x)

#         x+=skip
#         return x


# class Xception(nn.Module):
#     """
#     Xception optimized for the ImageNet dataset, as specified in
#     https://arxiv.org/pdf/1610.02357.pdf
#     """
#     def __init__(self, os):
#         """ Constructor
#         Args:
#             num_classes: number of classes
#         """
#         super(Xception, self).__init__()

#         stride_list = None
#         if os == 8:
#             stride_list = [2,1,1]
#         elif os == 16:
#             stride_list = [2,2,1]
#         else:
#             raise ValueError('xception.py: output stride=%d is not supported.'%os) 
#         self.conv1 = nn.Conv2d(3, 32, 3, 2, 1, bias=False)
#         self.bn1 = SynchronizedBatchNorm2d(32, momentum=bn_mom)
#         self.relu = nn.ReLU(inplace=True)
        
#         self.conv2 = nn.Conv2d(32,64,3,1,1,bias=False)
#         self.bn2 = SynchronizedBatchNorm2d(64, momentum=bn_mom)
#         #do relu here

#         self.block1=Block(64,128,2)
#         self.block2=Block(128,256,stride_list[0],inplace=False)
#         self.block3=Block(256,728,stride_list[1])

#         rate = 16//os
#         self.block4=Block(728,728,1,atrous=rate)
#         self.block5=Block(728,728,1,atrous=rate)
#         self.block6=Block(728,728,1,atrous=rate)
#         self.block7=Block(728,728,1,atrous=rate)

#         self.block8=Block(728,728,1,atrous=rate)
#         self.block9=Block(728,728,1,atrous=rate)
#         self.block10=Block(728,728,1,atrous=rate)
#         self.block11=Block(728,728,1,atrous=rate)

#         self.block12=Block(728,728,1,atrous=rate)
#         self.block13=Block(728,728,1,atrous=rate)
#         self.block14=Block(728,728,1,atrous=rate)
#         self.block15=Block(728,728,1,atrous=rate)

#         self.block16=Block(728,728,1,atrous=[1*rate,1*rate,1*rate])
#         self.block17=Block(728,728,1,atrous=[1*rate,1*rate,1*rate])
#         self.block18=Block(728,728,1,atrous=[1*rate,1*rate,1*rate])
#         self.block19=Block(728,728,1,atrous=[1*rate,1*rate,1*rate])
        
#         self.block20=Block(728,1024,stride_list[2],atrous=rate,grow_first=False)
#         #self.block12=Block(728,1024,2,2,start_with_relu=True,grow_first=False)

#         self.conv3 = SeparableConv2d(1024,1536,3,1,1*rate,dilation=rate,activate_first=False)
#         # self.bn3 = SynchronizedBatchNorm2d(1536, momentum=bn_mom)

#         self.conv4 = SeparableConv2d(1536,1536,3,1,1*rate,dilation=rate,activate_first=False)
#         # self.bn4 = SynchronizedBatchNorm2d(1536, momentum=bn_mom)

#         #do relu here
#         self.conv5 = SeparableConv2d(1536,2048,3,1,1*rate,dilation=rate,activate_first=False)
#         # self.bn5 = SynchronizedBatchNorm2d(2048, momentum=bn_mom)
#         self.layers = []

#         #------- init weights --------
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#             elif isinstance(m, SynchronizedBatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#         #-----------------------------

#     def forward(self, input):
#         self.layers = []
#         x = self.conv1(input)
#         x = self.bn1(x)
#         x = self.relu(x)
#         #self.layers.append(x)
#         x = self.conv2(x)
#         x = self.bn2(x)
#         x = self.relu(x)
        
#         x = self.block1(x)
#         x = self.block2(x)
#         self.layers.append(self.block2.hook_layer)
#         x = self.block3(x)
#         # self.layers.append(self.block3.hook_layer)
#         x = self.block4(x)
#         x = self.block5(x)
#         x = self.block6(x)
#         x = self.block7(x)
#         x = self.block8(x)
#         x = self.block9(x)
#         x = self.block10(x)
#         x = self.block11(x)
#         x = self.block12(x)
#         x = self.block13(x)
#         x = self.block14(x)
#         x = self.block15(x)
#         x = self.block16(x)
#         x = self.block17(x)
#         x = self.block18(x)
#         x = self.block19(x)
#         x = self.block20(x)       
#         # self.layers.append(self.block20.hook_layer)

#         x = self.conv3(x)
#         # x = self.bn3(x)
#         # x = self.relu(x)

#         x = self.conv4(x)
#         # x = self.bn4(x)
#         # x = self.relu(x)
        
#         x = self.conv5(x)
#         # x = self.bn5(x)
#         # x = self.relu(x)
#         self.layers.append(x)

#         return x

#     def get_layers(self):
#         return self.layers

#     def load_param(self, model_path):
#         param_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
#         if 'state_dict' in param_dict.keys():
#             param_dict = param_dict['state_dict']

        
#         start_with_module = False
#         for k in param_dict.keys():
#             if k.startswith('module.'):
#                 start_with_module = True
#                 break
#         if start_with_module:
#             param_dict = {k[7:] : v for k, v in param_dict.items() }
  
#         print('ignore_param:')
#         print([k for k, v in param_dict.items() if k not in self.state_dict() or self.state_dict()[k].size() != v.size()])
#         print('unload_param:')
#         print([k for k, v in self.state_dict().items() if k not in param_dict.keys() or param_dict[k].size() != v.size()] )

#         param_dict = {k: v for k, v in param_dict.items() if k in self.state_dict() and self.state_dict()[k].size() == v.size()}
#         for i in param_dict:
#             self.state_dict()[i].copy_(param_dict[i])

# def xception(pretrained=True, os=16):
#     model = Xception(os=os)
#     if pretrained:
#         old_dict = torch.load(model_urls['xception'])
#         # old_dict = model_zoo.load_url(model_urls['xception'])
#         # for name, weights in old_dict.items():
#         #     if 'pointwise' in name:
#         #         old_dict[name] = weights.unsqueeze(-1).unsqueeze(-1)
#         model_dict = model.state_dict()
#         old_dict = {k: v for k,v in old_dict.items() if ('itr' not in k and 'tmp' not in k and 'track' not in k)}
#         model_dict.update(old_dict)
        
#         model.load_state_dict(model_dict) 

#     return model





import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['Enc', 'FCAttention', 'Xception65', 'Xception71', 'get_xception', 'get_xception_71', 'get_xception_a']


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, bias=False, norm_layer=None):
        super(SeparableConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, 0, dilation, groups=in_channels,
                               bias=bias)
        self.bn = norm_layer(in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=bias)

    def forward(self, x):
        x = self.fix_padding(x, self.kernel_size, self.dilation)
        x = self.conv1(x)
        x = self.bn(x)
        x = self.pointwise(x)

        return x

    def fix_padding(self, x, kernel_size, dilation):
        kernel_size_effective = kernel_size + (kernel_size - 1) * (dilation - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        padded_inputs = F.pad(x, (pad_beg, pad_end, pad_beg, pad_end))
        return padded_inputs


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, reps, stride=1, dilation=1, norm_layer=None,
                 start_with_relu=True, grow_first=True, is_last=False):
        super(Block, self).__init__()
        if out_channels != in_channels or stride != 1:
            self.skip = nn.Conv2d(in_channels, out_channels, 1, stride, bias=False)
            self.skipbn = norm_layer(out_channels)
        else:
            self.skip = None
        self.relu = nn.ReLU(True)
        rep = list()
        filters = in_channels
        if grow_first:
            if start_with_relu:
                rep.append(self.relu)
            rep.append(SeparableConv2d(in_channels, out_channels, 3, 1, dilation, norm_layer=norm_layer))
            rep.append(norm_layer(out_channels))
            filters = out_channels
        for i in range(reps - 1):
            if grow_first or start_with_relu:
                rep.append(self.relu)
            rep.append(SeparableConv2d(filters, filters, 3, 1, dilation, norm_layer=norm_layer))
            rep.append(norm_layer(filters))
        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_channels, out_channels, 3, 1, dilation, norm_layer=norm_layer))
        if stride != 1:
            rep.append(self.relu)
            rep.append(SeparableConv2d(out_channels, out_channels, 3, stride, norm_layer=norm_layer))
            rep.append(norm_layer(out_channels))
        elif is_last:
            rep.append(self.relu)
            rep.append(SeparableConv2d(out_channels, out_channels, 3, 1, dilation, norm_layer=norm_layer))
            rep.append(norm_layer(out_channels))
        self.rep = nn.Sequential(*rep)

    def forward(self, x):
        out = self.rep(x)
        if self.skip is not None:
            skip = self.skipbn(self.skip(x))
        else:
            skip = x
        out = out + skip
        return out


class Xception65(nn.Module):
    """Modified Aligned Xception
    """

    def __init__(self, num_classes=1000, output_stride=32, norm_layer=nn.BatchNorm2d, inChannel=3):
        super(Xception65, self).__init__()
        if output_stride == 32:
            entry_block3_stride = 2
            exit_block20_stride = 2
            middle_block_dilation = 1
            exit_block_dilations = (1, 1)
        elif output_stride == 16:
            entry_block3_stride = 2
            exit_block20_stride = 1
            middle_block_dilation = 1
            exit_block_dilations = (1, 2)
        elif output_stride == 8:
            entry_block3_stride = 1
            exit_block20_stride = 1
            middle_block_dilation = 2
            exit_block_dilations = (2, 4)
        else:
            raise NotImplementedError
        # Entry flow
        self.conv1 = nn.Conv2d(inChannel, 32, 3, 2, 1, bias=False)
        self.bn1 = norm_layer(32)
        self.relu = nn.ReLU(True)

        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1, bias=False)
        self.bn2 = norm_layer(64)

        self.block1 = Block(64, 128, reps=2, stride=2, norm_layer=norm_layer, start_with_relu=False)
        self.block2 = Block(128, 256, reps=2, stride=2, norm_layer=norm_layer, start_with_relu=False, grow_first=True)
        self.block3 = Block(256, 728, reps=2, stride=entry_block3_stride, norm_layer=norm_layer,
                            start_with_relu=True, grow_first=True, is_last=True)

        # Middle flow
        midflow = list()
        for i in range(4, 20):
            midflow.append(Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation, norm_layer=norm_layer,
                                 start_with_relu=True, grow_first=True))
        self.midflow = nn.Sequential(*midflow)

        # Exit flow
        self.block20 = Block(728, 1024, reps=2, stride=exit_block20_stride, dilation=exit_block_dilations[0],
                             norm_layer=norm_layer, start_with_relu=True, grow_first=False, is_last=True)
        self.conv3 = SeparableConv2d(1024, 1536, 3, 1, dilation=exit_block_dilations[1], norm_layer=norm_layer)
        self.bn3 = norm_layer(1536)
        self.conv4 = SeparableConv2d(1536, 1536, 3, stride=1, dilation=exit_block_dilations[1], norm_layer=norm_layer)
        self.bn4 = norm_layer(1536)
        self.conv5 = SeparableConv2d(1536, 2048, 3, 1, dilation=exit_block_dilations[1], norm_layer=norm_layer)
        self.bn5 = norm_layer(2048)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        # Entry flow
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.block1(x)
        x = self.relu(x)
        # c1 = x
        x = self.block2(x)
        # c2 = x
        x = self.block3(x)

        # Middle flow
        x = self.midflow(x)
        # c3 = x

        # Exit flow
        x = self.block20(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)

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


class Xception71(nn.Module):
    """Modified Aligned Xception
    """

    def __init__(self, num_classes=1000, output_stride=32, norm_layer=nn.BatchNorm2d):
        super(Xception71, self).__init__()
        if output_stride == 32:
            entry_block3_stride = 2
            exit_block20_stride = 2
            middle_block_dilation = 1
            exit_block_dilations = (1, 1)
        elif output_stride == 16:
            entry_block3_stride = 2
            exit_block20_stride = 1
            middle_block_dilation = 1
            exit_block_dilations = (1, 2)
        elif output_stride == 8:
            entry_block3_stride = 1
            exit_block20_stride = 1
            middle_block_dilation = 2
            exit_block_dilations = (2, 4)
        else:
            raise NotImplementedError
        # Entry flow
        self.conv1 = nn.Conv2d(3, 32, 3, 2, 1, bias=False)
        self.bn1 = norm_layer(32)
        self.relu = nn.ReLU(True)

        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1, bias=False)
        self.bn2 = norm_layer(64)

        self.block1 = Block(64, 128, reps=2, stride=2, norm_layer=norm_layer, start_with_relu=False)
        self.block2 = nn.Sequential(
            Block(128, 256, reps=2, stride=2, norm_layer=norm_layer, start_with_relu=False, grow_first=True),
            Block(256, 728, reps=2, stride=2, norm_layer=norm_layer, start_with_relu=False, grow_first=True))
        self.block3 = Block(728, 728, reps=2, stride=entry_block3_stride, norm_layer=norm_layer,
                            start_with_relu=True, grow_first=True, is_last=True)

        # Middle flow
        midflow = list()
        for i in range(4, 20):
            midflow.append(Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation, norm_layer=norm_layer,
                                 start_with_relu=True, grow_first=True))
        self.midflow = nn.Sequential(*midflow)

        # Exit flow
        self.block20 = Block(728, 1024, reps=2, stride=exit_block20_stride, dilation=exit_block_dilations[0],
                             norm_layer=norm_layer, start_with_relu=True, grow_first=False, is_last=True)
        self.conv3 = SeparableConv2d(1024, 1536, 3, 1, dilation=exit_block_dilations[1], norm_layer=norm_layer)
        self.bn3 = norm_layer(1536)
        self.conv4 = SeparableConv2d(1536, 1536, 3, stride=1, dilation=exit_block_dilations[1], norm_layer=norm_layer)
        self.bn4 = norm_layer(1536)
        self.conv5 = SeparableConv2d(1536, 2048, 3, 1, dilation=exit_block_dilations[1], norm_layer=norm_layer)
        self.bn5 = norm_layer(2048)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        # Entry flow
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.block1(x)
        x = self.relu(x)
        # c1 = x
        x = self.block2(x)
        # c2 = x
        x = self.block3(x)

        # Middle flow
        x = self.midflow(x)
        # c3 = x

        # Exit flow
        x = self.block20(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)

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


# -------------------------------------------------
#                   For DFANet
# -------------------------------------------------
class BlockA(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dilation=1, norm_layer=None, start_with_relu=True):
        super(BlockA, self).__init__()
        if out_channels != in_channels or stride != 1:
            self.skip = nn.Conv2d(in_channels, out_channels, 1, stride, bias=False)
            self.skipbn = norm_layer(out_channels)
        else:
            self.skip = None
        self.relu = nn.ReLU(True)
        rep = list()
        inter_channels = out_channels // 4

        if start_with_relu:
            rep.append(self.relu)
        rep.append(SeparableConv2d(in_channels, inter_channels, 3, 1, dilation, norm_layer=norm_layer))
        rep.append(norm_layer(inter_channels))

        rep.append(self.relu)
        rep.append(SeparableConv2d(inter_channels, inter_channels, 3, 1, dilation, norm_layer=norm_layer))
        rep.append(norm_layer(inter_channels))

        if stride != 1:
            rep.append(self.relu)
            rep.append(SeparableConv2d(inter_channels, out_channels, 3, stride, norm_layer=norm_layer))
            rep.append(norm_layer(out_channels))
        else:
            rep.append(self.relu)
            rep.append(SeparableConv2d(inter_channels, out_channels, 3, 1, norm_layer=norm_layer))
            rep.append(norm_layer(out_channels))
        self.rep = nn.Sequential(*rep)

    def forward(self, x):
        out = self.rep(x)
        if self.skip is not None:
            skip = self.skipbn(self.skip(x))
        else:
            skip = x
        out = out + skip
        return out
    
    


class Enc(nn.Module):
    def __init__(self, in_channels, out_channels, blocks, norm_layer=None):
        super(Enc, self).__init__()
        block = list()
        block.append(BlockA(in_channels, out_channels, 2, norm_layer=norm_layer))
        for i in range(blocks - 1):
            block.append(BlockA(out_channels, out_channels, 1, norm_layer=norm_layer))
        self.block = nn.Sequential(*block)

    def forward(self, x):
        return self.block(x)


class FCAttention(nn.Module):
    def __init__(self, in_channels, norm_layer=None):
        super(FCAttention, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_channels, 1000)
        self.conv = nn.Sequential(
            nn.Conv2d(1000, in_channels, 1, bias=False),
            norm_layer(in_channels),
            nn.ReLU(True))

    def forward(self, x):
        n, c, _, _ = x.size()
        att = self.avgpool(x).view(n, c)
        att = self.fc(att).view(n, 1000, 1, 1)
        att = self.conv(att)
        return x * att.expand_as(x)


class XceptionA(nn.Module):
    def __init__(self, num_classes=1000, norm_layer=nn.BatchNorm2d):
        super(XceptionA, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3, 8, 3, 2, 1, bias=False),
                                   norm_layer(8),
                                   nn.ReLU(True))

        self.enc2 = Enc(8, 48, 4, norm_layer=norm_layer)
        self.enc3 = Enc(48, 96, 6, norm_layer=norm_layer)
        self.enc4 = Enc(96, 192, 4, norm_layer=norm_layer)

        self.fca = FCAttention(192, norm_layer=norm_layer)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(192, num_classes)

    def forward(self, x):
        x = self.conv1(x)

        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)
        x = self.fca(x)

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


# Constructor
def get_xception(pretrained=False, root='~/.torch/models', **kwargs):
    model = Xception65(**kwargs)
    if pretrained:
        from ..model_store import get_model_file
        model.load_state_dict(torch.load(get_model_file('xception', root=root)))
    return model


def get_xception_71(pretrained=False, root='~/.torch/models', **kwargs):
    model = Xception71(**kwargs)
    if pretrained:
        from ..model_store import get_model_file
        model.load_state_dict(torch.load(get_model_file('xception71', root=root)))
    return model


def get_xception_a(pretrained=False, root='~/.torch/models', **kwargs):
    model = XceptionA(**kwargs)
    if pretrained:
        from ..model_store import get_model_file
        model.load_state_dict(torch.load(get_model_file('xception_a', root=root)))
    return model


if __name__ == '__main__':
    model = get_xception_a()