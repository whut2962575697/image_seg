# -*- encoding: utf-8 -*-
'''
@File    :   __init__.py.py    
@Contact :   whut.hexin@foxmail.com
@License :   (C)Copyright 2017-2018, HeXin

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2020/1/28 13:21   xin      1.0         None
'''

import torchvision.transforms as T
from torch.utils.data.dataloader import DataLoader

from .data import BaseDataImageSet
from .custom_transform import Compose as Custom_Compose, Scale as Custom_Scale, \
    RandomScaleCrop as Custom_RandomScaleCrop, RandomHorizontallyFlip as Custom_RandomHorizontallyFlip, \
    RandomRotate as Custom_RandomRotate
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_trm(cfg, is_train=True):

    if is_train:
        main_transform = A.Compose(
    [
        A.RandomResizedCrop(cfg.INPUT.SIZE_TRAIN[0], cfg.INPUT.SIZE_TRAIN[1], p=1),
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=0.5),
        A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
        A.CoarseDropout(),
        A.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
        ToTensorV2(),
    ]
       )
    else:
        main_transform = A.Compose(
    [
        A.Resize(cfg.INPUT.SIZE_TRAIN[0], cfg.INPUT.SIZE_TRAIN[1]), 
        A.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD), 
        ToTensorV2()
    ]
        )

    return main_transform


def make_dataloader(cfg, num_gpus):
    train_main_transform = get_trm(cfg)
    val_main_transform = get_trm(cfg, False)
    num_workers = cfg.DATALOADER.NUM_WORKERS * num_gpus
    train_dataset = BaseDataImageSet(cfg, mode='train', img_suffix=cfg.DATASETS.IMG_SUFFIX, seg_map_suffix=cfg.DATASETS.SEG_MAP_SUFFIX,  main_transform=train_main_transform
                                     )
    val_dataset = BaseDataImageSet(cfg, mode='val', img_suffix=cfg.DATASETS.IMG_SUFFIX, seg_map_suffix=cfg.DATASETS.SEG_MAP_SUFFIX,  main_transform=val_main_transform
                                   )

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.SOLVER.PER_BATCH, shuffle=True,
        num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.SOLVER.PER_BATCH, shuffle=False,
        num_workers=num_workers
    )
    return train_loader, val_loader
