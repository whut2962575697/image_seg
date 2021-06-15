#!/usr/bin/env python
# coding: utf-8
#
# Author:   Kazuto Nakashima
# URL:      http://kazuto1011.github.io
# Created:  2018-03-26

import torch
import torch.nn as nn
import torch.nn.functional as F


class MSC(nn.Module):
    """
    Multi-scale inputs
    """

    def __init__(self, base, scales=None):
        super(MSC, self).__init__()
        self.base = base
        if scales:
            self.scales = scales
        else:
            self.scales = [0.5, 0.75]

    def forward(self, x):
        # Original
        logits = self.base(x)
        if isinstance(logits, list) or isinstance(logits, tuple):
            _, _, H, W = logits[0].shape
            interp = lambda l: F.interpolate(l, size=(H, W), mode="bilinear", align_corners=False)
            new_logits_pyramid = [[]]*len(logits)
            for p in self.scales:
                h = F.interpolate(x, scale_factor=p, mode="bilinear", align_corners=False)
                pyramid_res = self.base(h)
                for i, pyramid_item in enumerate(pyramid_res):
                    new_logits_pyramid[i].append(pyramid_item)
                

            # Pixel-wise max
            #logits_all = [logits] + [interp(l) for l in logits_pyramid]
            #logits_max = torch.max(torch.stack(logits_all), dim=0)[0]
            logits_all = []
            logits_max = []
            for _logits, logits_pyramid in zip(logits, new_logits_pyramid):
                _logits_all = [_logits] + [interp(l) for l in logits_pyramid]
                _logits_max = torch.max(torch.stack(_logits_all), dim=0)[0]
                logits_all.append([_logits]+logits_pyramid+[_logits_max])
                logits_max.append(_logits_max)
            if self.training:
                return logits_all
            else:
                return logits_max
        else:
            _, _, H, W = logits.shape
            interp = lambda l: F.interpolate(l, size=(H, W), mode="bilinear", align_corners=False)
            # Scaled
            logits_pyramid = []
            # Pixel-wise max
            logits_all = [logits] + [interp(l) for l in logits_pyramid]
            logits_max = torch.max(torch.stack(logits_all), dim=0)[0]
            if self.training:
                return [logits_max]+[logits] + logits_pyramid
            else:
                return logits_max

        


        