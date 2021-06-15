# -*- encoding: utf-8 -*-
'''
@File    :   __init__.py.py    
@Contact :   whut.hexin@foxmail.com
@License :   (C)Copyright 2017-2018, HeXin

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2020/1/28 13:18   xin      1.0         None
'''

from .unet import Unet
from .res_unet import Res_Unet
from .mf_unet import MfUnet


def build_model(cfg):
    if cfg.MODEL.NAME == "unet":
        model = Unet(cfg.MODEL.N_CHANNEL, cfg.MODEL.N_CLASS, cfg.MODEL.BILINEAR, cfg.MODEL.ENCODE_DIM, cfg.MODEL.DROPOUT)
    elif cfg.MODEL.NAME == "res_unet":
        model = Res_Unet(cfg.MODEL.N_CHANNEL, cfg.MODEL.N_CLASS, cfg.MODEL.DROPOUT)
    elif cfg.MODEL.NAME == "mf_unet":
        model = MfUnet(cfg.MODEL.N_CHANNEL, cfg.MODEL.N_CLASS, cfg.MODEL.BILINEAR, cfg.MODEL.ENCODE_DIM,
                     cfg.MODEL.DROPOUT)
    return model
