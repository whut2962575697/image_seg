# -*- encoding: utf-8 -*-
'''
@File    :   __init__.py.py    
@Contact :   whut.hexin@foxmail.com
@License :   (C)Copyright 2017-2018, HeXin

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2020/1/28 13:36   xin      1.0         None
'''

from .metrics import AvgerageMeter, accuracy, confusion_matrix
from .metrics import evaluate as m_evaluate
from .logging import setup_logger
from .mixup import mixup_data, mixup_criterion, mixup_data_multi_label
from .cut_mix import rand_bbox
from common.optimizer.ranger import Ranger
from common.optimizer.sgd_gc import SGD as SGD_GC

from sklearn.metrics import f1_score, accuracy_score
import torch
import numpy as np


from common.warmup import WarmupMultiStepLR, GradualWarmupScheduler
from common.poly import PolynomialLRDecay
from common.CyclicLR import CyclicCosAnnealingLR
from torch.optim.lr_scheduler import StepLR, MultiStepLR


# def make_optimizer(model,opt, lr,weight_decay, momentum=0.9,nesterov=True, bias_lr_factor=2, large_fc_lr=True):
#     base_lr = lr
#     params = []
#     for key, value in model.named_parameters():
#         if not value.requires_grad:
#             continue
#         lr = base_lr
#         if "bias" in key:
#             lr = base_lr * bias_lr_factor
#         if large_fc_lr:
#             if "dblock" in key or "decoder" in key or 'final' in key or "classifier" in key or "arcface" in key:
#                 lr = base_lr * 2
#                 print('Using two times learning rate for fc ')
#         params += [{"params": [value], "lr": lr, "initial_lr": lr, "weight_decay": weight_decay}]
#     if opt == 'SGD':
#         optimizer = getattr(torch.optim, opt)(params, momentum=momentum,nesterov=nesterov)
#     elif opt == 'AMSGRAD':
#         optimizer = getattr(torch.optim,'Adam')(params,amsgrad=True)
#     elif opt == 'Ranger':
#         optimizer = Ranger(params)
#     else:
#         optimizer = getattr(torch.optim, opt)(params)
#     return optimizer


def make_lr_scheduler(optimizer, scheduler_name, use_warmup=False, iter_num=120, t_max=120, steps=[40, 70],
                      milestones=[1], gamma=0.1, warmup_iter_num=0, min_lr=1e-4):
    assert scheduler_name in ['multi_step', 'cosine_annealing', 'poly', 'cyclic_linear']
    if scheduler_name == 'multi_step':
        scheduler = MultiStepLR(optimizer, steps, gamma)
    elif scheduler_name == 'cosine_annealing':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, t_max, eta_min=min_lr)
    elif scheduler_name == 'poly':
        scheduler = PolynomialLRDecay(optimizer, iter_num, end_learning_rate=min_lr, power=gamma)
    elif scheduler_name == 'cyclic_linear':
        scheduler = CyclicCosAnnealingLR(optimizer, milestones=milestones, eta_min=min_lr)
    else:
        scheduler = None
    if use_warmup:
        scheduler = GradualWarmupScheduler(optimizer, multiplier=10, total_epoch=warmup_iter_num,
                                           after_scheduler=scheduler)
    return scheduler


def make_optimizer(model, opt, lr, weight_decay, momentum=0.9, nesterov=True):
    # base_lr = lr
    # params = []
    # for key, value in model.named_parameters():
    #     if not value.requires_grad:
    #         continue
    #     lr = base_lr
        
    #     if "base_layer" not in key and 'encoder' not in key:
    #             lr = base_lr * 10
    #             print('Using 10 times learning rate for fc ')
    #     params += [{"params": [value], "lr": lr, "initial_lr": lr, "weight_decay": weight_decay}]
    if opt == 'SGD':
        optimizer = getattr(torch.optim, opt)(model.parameters(),lr=lr,weight_decay=weight_decay, momentum=momentum,nesterov=nesterov)
        # optimizer = getattr(torch.optim, opt)(params, momentum=momentum,nesterov=nesterov)
    elif opt == 'AMSGRAD':
        optimizer = getattr(torch.optim,'Adam')(model.parameters(),lr=lr,weight_decay=weight_decay,amsgrad=True)
        # optimizer = getattr(torch.optim, 'Adam')(model.parameters(), lr=lr, amsgrad=True)
    elif opt == 'Ranger':
        optimizer = Ranger(params=filter(lambda p: p.requires_grad, model.parameters()),
                       lr=lr)
    elif opt == 'SGD_GC':
        optimizer = SGD_GC(model.parameters(),lr=lr,weight_decay=weight_decay, momentum=momentum,use_gc=True, gc_conv_only=False)
    else:
        optimizer = getattr(torch.optim, opt)(model.parameters(),lr=lr,weight_decay=weight_decay)
    return optimizer


def calculate_score(cfg, outputs, targets, metric_type='train'):
    if cfg.MODEL.NAME == 'unet' or cfg.MODEL.NAME == 'res_unet' or cfg.MODEL.NAME == 'hrnet'  or cfg.MODEL.NAME == 'res_unet_3plus'  or cfg.MODEL.NAME == 'mf_unet' or cfg.MODEL.NAME == 'efficient_unet' or cfg.MODEL.NAME == 'dlinknet' or cfg.MODEL.NAME == 'deeplab_v3_plus':

        if metric_type=='train':
            f1 = f1_score(
            targets.flatten(),
            outputs.flatten(),
            average="macro")
            acc = accuracy_score(outputs.flatten(), targets.flatten())
            return f1, acc
        else:
            conf_mat = confusion_matrix(pred=outputs.flatten(),
                                                label=targets.flatten(),
                                                num_classes=cfg.MODEL.N_CLASS)
            # acc, acc_per_class, acc_cls, IoU, mean_IoU, kappa = evaluate(conf_mat)
            # return f1, acc, acc_per_class, acc_cls, IoU, mean_IoU, kappa
            return conf_mat


class AddDepthChannels:
    def __call__(self, tensor):
        _, h, w = tensor.size()
        for row, const in enumerate(np.linspace(0, 1, h)):
            tensor[1, row, :] = const
        tensor[2] = tensor[0] * tensor[1]
        return tensor

    def __repr__(self):
        return self.__class__.__name__