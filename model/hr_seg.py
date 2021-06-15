import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F

from .modules.backbones.hrnet import hrnetv2_18, hrnetv2_32, hrnetv2_48


class HRNet(nn.Module):
    def __init__(self, in_channels, n_classes, backbone, model_path, dropout_rate=0.0):
        super().__init__()

        if backbone == 'hrnetv2_18':

            self.base_model = hrnetv2_18(pretrained=False)

        elif backbone == 'hrnetv2_32':

            self.base_model = hrnetv2_32(pretrained=False)

        elif backbone == 'hrnetv2_48':

            self.base_model = hrnetv2_48(pretrained=False)

        if model_path:
            self.base_model.load_param(model_path)

        # input_dim >3
        if in_channels > 3:
            with torch.no_grad():
                pretrained_conv1 = self.base_model.conv1.weight.clone()
                self.base_model.conv1 = torch.nn.Conv2d(in_channels, 64, 3, 2, 1, bias=False)
                torch.nn.init.kaiming_normal_(
                    self.base_model.conv1.weight, mode='fan_out', nonlinearity='relu')
                # Re-assign pretraiend weights to first 3 channels
                # (assuming alpha channel is last in your input data)
                self.base_model.conv1.weight[:, :3] = pretrained_conv1

        last_inp_channels = np.int(np.sum(self.base_model.final_stage_channels))

        self.last_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=last_inp_channels,
                out_channels=last_inp_channels,
                kernel_size=1,
                stride=1,
                padding=0),
            nn.BatchNorm2d(last_inp_channels, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=last_inp_channels,
                out_channels=n_classes,
                kernel_size=1,
                stride=1,
                padding=0)
        )
        self.use_dropout = False
        if dropout_rate > 0:
            self.use_dropout = True
            self.dropout = nn.Dropout2d(dropout_rate)

    def freeze_bn(self):
        print("freeze bacth normalization successfully!")
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.InstanceNorm2d):
                m.eval()

    def forward(self, input):
        x = self.base_model.conv1(input)
        x = self.base_model.bn1(x)
        x = self.base_model.relu(x)
        x = self.base_model.conv2(x)
        x = self.base_model.bn2(x)
        x = self.base_model.relu(x)
        x = self.base_model.layer1(x)

        x_list = []
        for i in range(self.base_model.stage2_cfg['NUM_BRANCHES']):
            if self.base_model.transition1[i] is not None:
                x_list.append(self.base_model.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.base_model.stage2(x_list)

        x_list = []
        for i in range(self.base_model.stage3_cfg['NUM_BRANCHES']):
            if self.base_model.transition2[i] is not None:
                if i < self.base_model.stage2_cfg['NUM_BRANCHES']:
                    x_list.append(self.base_model.transition2[i](y_list[i]))
                else:
                    x_list.append(self.base_model.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.base_model.stage3(x_list)

        x_list = []
        for i in range(self.base_model.stage4_cfg['NUM_BRANCHES']):
            if self.base_model.transition3[i] is not None:
                if i < self.base_model.stage3_cfg['NUM_BRANCHES']:
                    x_list.append(self.base_model.transition3[i](y_list[i]))
                else:
                    x_list.append(self.base_model.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        x = self.base_model.stage4(x_list)

        # Upsampling
        x0_h, x0_w = x[0].size(2), x[0].size(3)
        x1 = F.interpolate(x[1], size=(x0_h, x0_w), mode='bilinear', align_corners=False)
        x2 = F.interpolate(x[2], size=(x0_h, x0_w), mode='bilinear', align_corners=False)
        x3 = F.interpolate(x[3], size=(x0_h, x0_w), mode='bilinear', align_corners=False)

        x = torch.cat([x[0], x1, x2, x3], 1)

        if self.use_dropout:
            x = self.dropout(x)

        x = self.last_layer(x)
        x = F.interpolate(input=x, scale_factor=4, mode='bilinear', align_corners=False)

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