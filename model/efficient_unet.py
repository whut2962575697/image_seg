from collections import OrderedDict
from .modules.utils import *
from .modules.backbones.efficientnet import EfficientNet

from .modules.scse import cSE, sSE


__all__ = ['EfficientUnet', 'get_efficientunet_b0', 'get_efficientunet_b1', 'get_efficientunet_b2',
           'get_efficientunet_b3', 'get_efficientunet_b4', 'get_efficientunet_b5', 'get_efficientunet_b6',
           'get_efficientunet_b7']


def get_blocks_to_be_concat(model, x):
    shapes = set()
    blocks = OrderedDict()
    hooks = []
    count = 0

    def register_hook(module):

        def hook(module, input, output):
            try:
                nonlocal count
                if module.name == f'blocks_{count}_output_batch_norm':
                    count += 1
                    shape = output.size()[-2:]
                    if shape not in shapes:
                        shapes.add(shape)
                        blocks[module.name] = output

                elif module.name == 'head_swish':
                    # when module.name == 'head_swish', it means the program has already got all necessary blocks for
                    # concatenation. In my dynamic unet implementation, I first upscale the output of the backbone,
                    # (in this case it's the output of 'head_swish') concatenate it with a block which has the same
                    # Height & Width (image size). Therefore, after upscaling, the output of 'head_swish' has bigger
                    # image size. The last block has the same image size as 'head_swish' before upscaling. So we don't
                    # really need the last block for concatenation. That's why I wrote `blocks.popitem()`.
                    blocks.popitem()
                    blocks[module.name] = output

            except AttributeError:
                pass

        if (
                not isinstance(module, nn.Sequential)
                and not isinstance(module, nn.ModuleList)
                and not (module == model)
        ):
            hooks.append(module.register_forward_hook(hook))

    # register hook
    model.apply(register_hook)

    # make a forward pass to trigger the hooks
    model(x)

    # remove these hooks
    for h in hooks:
        h.remove()

    return blocks


class EfficientUnet(nn.Module):
    def __init__(self, in_channels, n_classes, model_name,  model_path, dropout_rate=0.5, concat_input=True, use_attention_block=False, use_scse=False):
        super().__init__()

        # self.encoder = encoder
        self.encoder = EfficientNet.encoder(model_name, model_path=model_path, in_channels=in_channels, drop_connect_rate=dropout_rate)
        self.concat_input = concat_input

        self.up_conv1 = up_conv(self.n_channels, 512)
        self.double_conv1 = conv_block(self.size[0], 512)
        self.up_conv2 = up_conv(512, 256)
        self.double_conv2 = conv_block(self.size[1], 256)
        self.up_conv3 = up_conv(256, 128)
        self.double_conv3 = conv_block(self.size[2], 128)
        self.up_conv4 = up_conv(128, 64)
        self.double_conv4 = conv_block(self.size[3], 64)
        
        self.use_attention_block = use_attention_block


        self.use_scse = use_scse
        if self.use_scse:
            self.spatial_gate1 = sSE(512)
            self.channel_gate1 = cSE(512)

            self.spatial_gate2 = sSE(256)
            self.channel_gate2 = cSE(256)


            self.spatial_gate3 = sSE(128)
            self.channel_gate3 = cSE(128)

            self.spatial_gate4 = sSE(64)
            self.channel_gate4 = cSE(64)

        if self.concat_input:
            self.up_conv_input = up_conv(64, 32)
            self.double_conv_input = conv_block(self.size[4], 32)

        self.final_conv = nn.Conv2d(self.size[5], n_classes, kernel_size=1)

    @property
    def n_channels(self):
        n_channels_dict = {'efficientnet-b0': 1280, 'efficientnet-b1': 1280, 'efficientnet-b2': 1408,
                           'efficientnet-b3': 1536, 'efficientnet-b4': 1792, 'efficientnet-b5': 2048,
                           'efficientnet-b6': 2304, 'efficientnet-b7': 2560}
        return n_channels_dict[self.encoder.name]

    @property
    def size(self):
        size_dict = {'efficientnet-b0': [592, 296, 152, 80, 35, 32], 'efficientnet-b1': [592, 296, 152, 80, 35, 32],
                     'efficientnet-b2': [600, 304, 152, 80, 35, 32], 'efficientnet-b3': [608, 304, 160, 88, 35, 32],
                     'efficientnet-b4': [624, 312, 160, 88, 35, 32], 'efficientnet-b5': [640, 320, 168, 88, 35, 32],
                     'efficientnet-b6': [656, 328, 168, 96, 35, 32], 'efficientnet-b7': [672, 336, 176, 96, 35, 32]}
        return size_dict[self.encoder.name]

    def forward(self, x):
        input_ = x

        blocks = get_blocks_to_be_concat(self.encoder, x)
        _, x = blocks.popitem()

        x = self.up_conv1(x)
        x4 = blocks.popitem()[1]
#         print(x4.shape)

        x = torch.cat([x, x4], dim=1)
        x = self.double_conv1(x)
        if self.use_scse:
            x = self.spatial_gate1(x)*x+self.channel_gate1(x)*x

        x = self.up_conv2(x)
        x3 = blocks.popitem()[1]
#         print(x3.shape)

        x = torch.cat([x, x3], dim=1)
        x = self.double_conv2(x)
        if self.use_scse:
            x = self.spatial_gate2(x)*x+self.channel_gate2(x)*x

        x = self.up_conv3(x)
        x2 = blocks.popitem()[1]
#         print(x2.shape)

        x = torch.cat([x, x2], dim=1)
        x = self.double_conv3(x)
        if self.use_scse:
            x = self.spatial_gate3(x)*x+self.channel_gate3(x)*x

        x = self.up_conv4(x)
        x1 = blocks.popitem()[1]
#         print(x1.shape)

        x = torch.cat([x, x1], dim=1)
        x = self.double_conv4(x)
        if self.use_scse:
            x = self.spatial_gate4(x)*x+self.channel_gate4(x)*x

        if self.concat_input:
            x = self.up_conv_input(x)
            x = torch.cat([x, input_], dim=1)
            x = self.double_conv_input(x)

        x = self.final_conv(x)

        return x