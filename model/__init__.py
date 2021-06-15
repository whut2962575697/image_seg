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

from .efficient_unet import EfficientUnet

from .hr_seg import HRNet
from .deeplab_v3_plus import DeepV3Plus
from .dlinknet import DLinkNet
from .hr_seg import HRNet

from .msc import MSC
from .msc_hierarchical import Basic
# from .double_res_unet import Double_Res_Unet


def build_model(cfg):
    if cfg.MODEL.NAME == "unet":
        model = Unet(cfg.MODEL.N_CHANNEL, cfg.MODEL.N_CLASS, cfg.MODEL.UNET.BILINEAR, cfg.MODEL.UNET.ENCODE_DIM, cfg.MODEL.DROPOUT, cfg.MODEL.UNET.SCSE, cfg.MODEL.UNET.DBLOCK, cfg.MODEL.UNET.ATTENTION_BLOCK, cfg.MODEL.UNET.RRCNN_BLOCK, cfg.MODEL.UNET.RRCNN_BLOCK_T)
    elif cfg.MODEL.NAME == "efficient_unet":
        model = EfficientUnet(cfg.MODEL.N_CHANNEL, cfg.MODEL.N_CLASS, cfg.MODEL.EFFICIENT_UNET.BACKBONE_NAME, cfg.MODEL.WEIGHT, cfg.MODEL.DROPOUT, cfg.MODEL.EFFICIENT_UNET.CONCAT_INPUT, cfg.MODEL.EFFICIENT_UNET.ATTENTION_BLOCK, cfg.MODEL.EFFICIENT_UNET.SCSE)
    elif cfg.MODEL.NAME == "deeplab_v3_plus":
        model = DeepV3Plus(cfg.MODEL.N_CHANNEL, cfg.MODEL.N_CLASS, cfg.MODEL.DEEPLAB_V3_PLUS.BACKBONE_NAME, cfg.MODEL.WEIGHT)
    elif cfg.MODEL.NAME == "dlinknet":
        model = DLinkNet(cfg.MODEL.N_CHANNEL, cfg.MODEL.N_CLASS, cfg.MODEL.DLINKNET.BACKBONE_NAME, cfg.MODEL.WEIGHT, cfg.MODEL.DLINKNET.SCSE, cfg.MODEL.DLINKNET.MISH)
    elif cfg.MODEL.NAME == "hrnet":
        model = HRNet(cfg.MODEL.N_CHANNEL, cfg.MODEL.N_CLASS, cfg.MODEL.HRNET.BACKBONE_NAME, cfg.MODEL.WEIGHT, cfg.MODEL.DROPOUT)
    elif cfg.MODEL.NAME == "res_unet":
        model = Res_Unet(cfg.MODEL.N_CHANNEL, cfg.MODEL.N_CLASS, cfg.MODEL.RES_UNET.BACKBONE_NAME, cfg.MODEL.WEIGHT, cfg.MODEL.DROPOUT, cfg.MODEL.RES_UNET.SCSE, cfg.MODEL.RES_UNET.MISH, cfg.MODEL.RES_UNET.DB_BLOCK, cfg.MODEL.RES_UNET.HYPERCOLUMN)

        if cfg.SOLVER.MULTI_SCALE:
            model = MSC(base=model, scales=cfg.SOLVER.MULTI_SCALE)
            # model = Basic(base=model)

    return model

