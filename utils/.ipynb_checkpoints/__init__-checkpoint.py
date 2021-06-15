# -*- encoding: utf-8 -*-
'''
@File    :   __init__.py.py    
@Contact :   whut.hexin@foxmail.com
@License :   (C)Copyright 2017-2018, HeXin

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2020/1/28 13:36   xin      1.0         None
'''

from .metrics import AvgerageMeter, accuracy, confusion_matrix, evaluate
from .logging import setup_logger
from .mixup import mixup_data, mixup_criterion
from common.optimizer.ranger import Ranger

from sklearn.metrics import f1_score, accuracy_score
import torch
import numpy as np



def make_optimizer(model, opt, lr, weight_decay, momentum=0.9, nesterov=True):
    if opt == 'SGD':
        optimizer = getattr(torch.optim, opt)(model.parameters(),lr=lr,weight_decay=weight_decay, momentum=momentum,nesterov=nesterov)
    elif opt == 'AMSGRAD':
        optimizer = getattr(torch.optim,'Adam')(model.parameters(),lr=lr,weight_decay=weight_decay,amsgrad=True)
        # optimizer = getattr(torch.optim, 'Adam')(model.parameters(), lr=lr, amsgrad=True)
    elif opt == 'Ranger':
        optimizer = Ranger(params=filter(lambda p: p.requires_grad, model.parameters()),
                       lr=lr)
    else:
        optimizer = getattr(torch.optim, opt)(model.parameters(),lr=lr,weight_decay=weight_decay)
    return optimizer




def calculate_score(cfg, outputs, targets, metric_type='train'):
    if cfg.MODEL.NAME == 'unet' or cfg.MODEL.NAME == 'res_unet' or cfg.MODEL.NAME == 'mf_unet':
#         acc = accuracy(outputs, targets, 1)
#         f1 = 0
#         targets = targets.view(-1)
#         outputs = torch.argmax(outputs, 1)
#         outputs = outputs.view(-1)
        f1 = f1_score(
            targets.flatten(),
            outputs.flatten(),
            average="macro")
        
        if metric_type=='train':
            acc = accuracy_score(outputs.flatten(), targets.flatten())
            return f1, acc
        else:
            conf_mat = confusion_matrix(pred=outputs.flatten(),
                                                label=targets.flatten(),
                                                num_classes=cfg.MODEL.N_CLASS)
            acc, acc_per_class, acc_cls, IoU, mean_IoU, kappa = evaluate(conf_mat)
            return f1, acc, acc_per_class, acc_cls, IoU, mean_IoU, kappa