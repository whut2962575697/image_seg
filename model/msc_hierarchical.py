"""
Copyright 2020 Nvidia Corporation
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
This is an alternative implementation of mscale, where we feed pairs of 
features from both lower and higher resolution images into the attention head.
"""
import torch
from torch import nn

# from network.utils import get_aspp, get_trunk
# from network.utils import make_seg_head, make_attn_head


from collections import OrderedDict

import torch
import torch.nn.functional as F

from torch import nn


class MscaleBase(nn.Module):
    """
    Multi-scale attention segmentation model base class
    """
    def __init__(self):
        super(MscaleBase, self).__init__()
        self.n_scales = []

    def _fwd(self, x):
        pass

    def nscale_forward(self, inputs, scales):
        """
        Hierarchical attention, primarily used for getting best inference
        results.
        We use attention at multiple scales, giving priority to the lower
        resolutions. For example, if we have 4 scales {0.5, 1.0, 1.5, 2.0},
        then evaluation is done as follows:
              p_joint = attn_1.5 * p_1.5 + (1 - attn_1.5) * down(p_2.0)
              p_joint = attn_1.0 * p_1.0 + (1 - attn_1.0) * down(p_joint)
              p_joint = up(attn_0.5 * p_0.5) * (1 - up(attn_0.5)) * p_joint
        The target scale is always 1.0, and 1.0 is expected to be part of the
        list of scales. When predictions are done at greater than 1.0 scale,
        the predictions are downsampled before combining with the next lower
        scale.
        Inputs:
          scales - a list of scales to evaluate
          inputs - dict containing 'images', the input, and 'gts', the ground
                   truth mask
        Output:
          If training, return loss, else return prediction + attention
        """
        x_1x = inputs

        assert 1.0 in scales, 'expected 1.0 to be the target scale'
        # Lower resolution provides attention for higher rez predictions,
        # so we evaluate in order: high to low
        scales = sorted(scales, reverse=True)
        pred = None
        last_feats = None

        for idx, s in enumerate(scales):
            # x = ResizeX(x_1x, s)
            x = torch.nn.functional.interpolate(x_1x, scale_factor=s, mode='bilinear', 
            align_corners=False, recompute_scale_factor=True)
            p, feats = self._fwd(x)

            # Generate attention prediction
            if idx > 0:
                assert last_feats is not None
                # downscale feats
                # last_feats = scale_as(last_feats, feats)
                last_feats = torch.nn.functional.interpolate(last_feats, size=(feats.size(2), feats.size(3)), mode='bilinear',align_corners=False)
                cat_feats = torch.cat([feats, last_feats], 1)
                attn = self.scale_attn(cat_feats)
                # attn = scale_as(attn, p)
                attn = torch.nn.functional.interpolate(attn, size=(p.size(2), p.size(3)), mode='bilinear',align_corners=False)

            if pred is None:
                # This is the top scale prediction
                pred = p
            elif s >= 1.0:
                # downscale previous
                # pred = scale_as(pred, p)
                pred = torch.nn.functional.interpolate(pred, size=(p.size(2), p.size(3)), mode='bilinear',align_corners=False)
                pred = attn * p + (1 - attn) * pred
            else:
                # upscale current
                p = attn * p
                # p = scale_as(p, pred)
                p = torch.nn.functional.interpolate(p, size=(pred.size(2), pred.size(3)), mode='bilinear',align_corners=False)
                # attn = scale_as(attn, pred)
                attn = torch.nn.functional.interpolate(attn, size=(pred.size(2), pred.size(3)), mode='bilinear',align_corners=False)
                pred = p + (1 - attn) * pred

            last_feats = feats
        return pred, attn
        # if self.training:
        #     assert 'gts' in inputs
        #     gts = inputs['gts']
        #     loss = self.criterion(pred, gts)
        #     return loss
        # else:
        #     # FIXME: should add multi-scale values for pred and attn
        #     return {'pred': pred,
        #             'attn_10x': attn}

    def two_scale_forward(self, inputs):


        x_1x = inputs
        # x_lo = ResizeX(x_1x, cfg.MODEL.MSCALE_LO_SCALE)
        x_lo = torch.nn.functional.interpolate(x_1x, scale_factor=0.5, mode='bilinear', 
            align_corners=False, recompute_scale_factor=True)

        p_lo, feats_lo = self._fwd(x_lo)
        p_1x, feats_hi = self._fwd(x_1x)

        # feats_hi = scale_as(feats_hi, feats_lo)
        feats_hi = torch.nn.functional.interpolate(feats_hi, size=(feats_lo.size(2), feats_lo.size(3)), mode='bilinear',align_corners=False)
        cat_feats = torch.cat([feats_lo, feats_hi], 1)
        logit_attn = self.scale_attn(cat_feats)
        # logit_attn = scale_as(logit_attn, p_lo)
        logit_attn = torch.nn.functional.interpolate(logit_attn, size=(p_lo.size(2), p_lo.size(3)), mode='bilinear',align_corners=False)

        p_lo = logit_attn * p_lo
        # p_lo = scale_as(p_lo, p_1x)
        p_lo = torch.nn.functional.interpolate(p_lo, size=(p_1x.size(2), p_1x.size(3)), mode='bilinear',align_corners=False)
        # logit_attn = scale_as(logit_attn, p_1x)
        logit_attn = torch.nn.functional.interpolate(logit_attn, size=(p_1x.size(2), p_1x.size(3)), mode='bilinear',align_corners=False)
        joint_pred = p_lo + (1 - logit_attn) * p_1x

        return joint_pred
        # if self.training:
        #     assert 'gts' in inputs
        #     gts = inputs['gts']
        #     loss = self.criterion(joint_pred, gts)
        #     return loss
        # else:
        #     # FIXME: should add multi-scale values for pred and attn
        #     return {'pred': joint_pred,
        #             'attn_10x': logit_attn}

    def forward(self, inputs):
        if self.n_scales and not self.training:
            return self.nscale_forward(inputs, self.n_scales)

        return self.two_scale_forward(inputs)

def init_attn(m):
    for module in m.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.zeros_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.5)
        elif isinstance(module, nn.BatchNorm2d):
            module.weight.data.fill_(1)
            module.bias.data.zero_()

def make_attn_head(in_ch, bot_ch, out_ch):
    attn = nn.Sequential(
        nn.Conv2d(in_ch, bot_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(bot_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(bot_ch, bot_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(bot_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(bot_ch, out_ch, kernel_size=out_ch, bias=False),
        nn.Sigmoid())

    init_attn(attn)
    return attn


class Basic(MscaleBase):
    """
    """
    def __init__(self, base=None):
        super(Basic, self).__init__()
       
        # self.backbone, _, _, high_level_ch = get_trunk(
        #     trunk_name=trunk, output_stride=8)

        # self.cls_head = make_seg_head(in_ch=high_level_ch, bot_ch=256,
        #                               out_ch=num_classes)
        self.base = base
        self.scale_attn = make_attn_head(in_ch=256 * 2, bot_ch=256,
                                         out_ch=1)

    def two_scale_forward(self, inputs):


        x_1x = inputs
        # x_lo = ResizeX(x_1x, cfg.MODEL.MSCALE_LO_SCALE)
        x_lo = torch.nn.functional.interpolate(x_1x, scale_factor=0.5, mode='bilinear', 
            align_corners=False, recompute_scale_factor=True)

        p_lo, feats_lo = self._fwd(x_lo)
        p_1x, feats_hi = self._fwd(x_1x)

        # feats_hi = scale_as(feats_hi, feats_lo)
        feats_hi = torch.nn.functional.interpolate(feats_hi, size=(feats_lo.size(2), feats_lo.size(3)), mode='bilinear',align_corners=False)
        cat_feats = torch.cat([feats_lo, feats_hi], 1)
        logit_attn = self.scale_attn(cat_feats)
        # logit_attn = scale_as(logit_attn, p_lo)
        logit_attn = torch.nn.functional.interpolate(logit_attn, size=(p_lo.size(2), p_lo.size(3)), mode='bilinear',align_corners=False)

        p_lo = logit_attn * p_lo
        # p_lo = scale_as(p_lo, p_1x)
        p_lo = torch.nn.functional.interpolate(p_lo, size=(p_1x.size(2), p_1x.size(3)), mode='bilinear',align_corners=False)
        # logit_attn = scale_as(logit_attn, p_1x)
        logit_attn = torch.nn.functional.interpolate(logit_attn, size=(p_1x.size(2), p_1x.size(3)), mode='bilinear',align_corners=False)
        joint_pred = p_lo + (1 - logit_attn) * p_1x

        return joint_pred
        # if self.training:
        #     assert 'gts' in inputs
        #     gts = inputs['gts']
        #     loss = self.criterion(joint_pred, gts)
        #     return loss
        # else:
        #     # FIXME: should add multi-scale values for pred and attn
        #     return {'pred': joint_pred,
        #             'attn_10x': logit_attn}
    def _fwd(self, x):
        pred, final_features = self.base(x)
        return pred, final_features