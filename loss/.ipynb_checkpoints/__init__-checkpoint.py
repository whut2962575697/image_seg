# -*- encoding: utf-8 -*-
'''
@File    :   __init__.py.py    
@Contact :   whut.hexin@foxmail.com
@License :   (C)Copyright 2017-2018, HeXin

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2020/1/28 13:19   xin      1.0         None
'''

import torch.nn as nn

from .class_blance_loss import CB_loss
from .cross_entropy_labelsmooth import CrossEntropyLabelSmooth
from .focal_loss import FocalLoss
from .lovasz_losses import lovasz_softmax


class Unet_Loss(nn.modules.loss._Loss):
    def __init__(self, device):
        super(Unet_Loss, self).__init__()
        self.device = device

    def forward(self, outputs, targets, loss_type='ce'):
        if loss_type == 'cb':
            outputs = outputs.permute(0,2,3,1)
            outputs = outputs.contiguous().view(-1, outputs.shape[-1])
            CE_Loss = CB_loss(targets.view(-1), outputs,
                              [13172662, 11065351, 5451002, 2564979, 12936544, 6174531, 601401, 964881, 2049229, 5192430, 38179], 11, 'focal', 0.999, 2.0)
        elif loss_type == 'ce':
            ce_loss = nn.CrossEntropyLoss(ignore_index=-1).to(self.device)
            CE_Loss = ce_loss(outputs, targets)
        elif loss_type == 'focal':
            ce_loss = FocalLoss(gamma=3).to(self.device)
            CE_Loss = ce_loss(outputs, targets)
        elif loss_type == 'ls':
            outputs = outputs.permute(0, 2, 3, 1)
            outputs = outputs.contiguous().view(-1, outputs.shape[-1])
            ce_loss = CrossEntropyLabelSmooth(11)
            CE_Loss = ce_loss(outputs, targets.view(-1))
        elif loss_type == 'ls_with_lovasz':
            LZ_Loss = lovasz_softmax(outputs, targets)
            outputs = outputs.permute(0, 2, 3, 1)
            outputs = outputs.contiguous().view(-1, outputs.shape[-1])
            ce_loss = CrossEntropyLabelSmooth(11)
            CE_Loss = ce_loss(outputs, targets.view(-1))
            CE_Loss = 0.5*CE_Loss + 0.5*LZ_Loss

        print('\r[loss] ce:%.2f\t ' % (CE_Loss.data.cpu().numpy(),), end=' ')
        return CE_Loss


def make_loss(cfg, device):
    if cfg.MODEL.NAME == 'unet' or cfg.MODEL.NAME == 'res_unet' or cfg.MODEL.NAME == 'mf_unet':
        loss = Unet_Loss(device)
    return loss