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

from .data import BaseDataImageSet, BaseTestDataImageSet
import albumentations as A
from .copy_paste import CopyPaste
import cv2
from albumentations.pytorch import ToTensorV2


from prefetch_generator import BackgroundGenerator


class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


def get_trm(cfg, is_train=True):

    if is_train:
        main_transform_list = [
            # A.RandomResizedCrop(cfg.INPUT.SIZE_TRAIN[0], cfg.INPUT.SIZE_TRAIN[1], scale=(0.5, 1.0), ratio=(0.9, 1.1), p=0.5),
            #  A.LongestMaxSize(cfg.INPUT.SIZE_TRAIN[0]),
            # A.Resize(cfg.INPUT.SIZE_TRAIN[0], cfg.INPUT.SIZE_TRAIN[1])
            # A.RandomScale(0.25, p=1)
            # A.RandomSizedCrop((cfg.INPUT.SIZE_TRAIN[0]-100, cfg.INPUT.SIZE_TRAIN[0]+100), cfg.INPUT.SIZE_TRAIN[0], cfg.INPUT.SIZE_TRAIN[1], w2h_ratio=2.0, p=1)
        ]
        # main_transform_list = [A.RandomSizedCrop((cfg.INPUT.SIZE_TRAIN[0]-50, cfg.INPUT.SIZE_TRAIN[1]+50), cfg.INPUT.SIZE_TRAIN[0], cfg.INPUT.SIZE_TRAIN[1], p=1)]
        if cfg.INPUT.USE_RESIZE:
            main_transform_list.append(A.Resize(cfg.INPUT.SIZE_TRAIN[0], cfg.INPUT.SIZE_TRAIN[1]))
        if cfg.INPUT.USE_COPY_PASTE:
            main_transform_list.append(A.RandomScale(scale_limit=(-0.9, 1), p=1))
            main_transform_list.append(A.PadIfNeeded(cfg.INPUT.SIZE_TRAIN[0], cfg.INPUT.SIZE_TRAIN[1], border_mode=0, value=0, mask_value=255))# pads with image in the center, not the top left like the paper
            main_transform_list.append(A.RandomCrop(cfg.INPUT.SIZE_TRAIN[0], cfg.INPUT.SIZE_TRAIN[1]))
            main_transform_list.append(CopyPaste(blend=True, sigma=1, pct_objects_paste=0.8, p=1.))
        if cfg.INPUT.USE_VFLIP:
            main_transform_list.append(A.VerticalFlip(p=0.5))
        if cfg.INPUT.USE_HFLIP:
            main_transform_list.append(A.HorizontalFlip(p=0.5))
        if cfg.INPUT.USE_RANDOMROTATE90:
            main_transform_list.append(A.RandomRotate90(p=0.5))
        if cfg.INPUT.USE_SHIFTSCALEROTATE:
            main_transform_list.append(A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=0, p=0.5))
        if cfg.INPUT.USE_HUESATURATIONVALUE:
            main_transform_list.append(A.HueSaturationValue(hue_shift_limit=30, sat_shift_limit=5, val_shift_limit=15, p=0.5))
        if cfg.INPUT.USE_RGBSHIFT:
            main_transform_list.append(A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.5))
        if cfg.INPUT.USE_RANDOMBRIGHTNESSCONTRAST:
            main_transform_list.append(A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5))
        if cfg.INPUT.USE_CLAHE:
            main_transform_list.append(A.CLAHE(p=0.5))
        if cfg.INPUT.USE_RANDOMGAMMA:
            main_transform_list.append(A.RandomGamma(p=0.5))
        if cfg.INPUT.USE_BLUR:
            main_transform_list.append(A.MotionBlur(p=0.5))
        if cfg.INPUT.USE_GAUSSNOISE:
            main_transform_list.append(A.GaussNoise(p=0.5))
        if cfg.INPUT.USE_ELASTICTRANSFORM:
            main_transform_list.append(A.ElasticTransform(p=0.5))
            # main_transform_list.append(A.OneOf([
            #     A.ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
            #     A.GridDistortion(p=0.5),
            #     A.OpticalDistortion(distort_limit=1, shift_limit=0.5, p=1)], p=0.8))
            
        if cfg.INPUT.USE_CUTOUT:
            main_transform_list.append(A.Cutout(p=0.5))
        if cfg.INPUT.USE_GRIDMASK:
            main_transform_list.append(A.GridDropout(p=0.5))
    else:
        main_transform_list = []
        if cfg.INPUT.USE_RESIZE:
            main_transform_list.append(A.Resize(cfg.INPUT.SIZE_TRAIN[0], cfg.INPUT.SIZE_TRAIN[1]))

    main_transform_list.append(A.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD))
    main_transform_list.append(ToTensorV2())
    main_transform = A.Compose(main_transform_list)

    return main_transform


def make_dataloader(cfg, num_gpus):
    train_main_transform = get_trm(cfg)
    val_main_transform = get_trm(cfg, False)
    num_workers = cfg.DATALOADER.NUM_WORKERS * num_gpus
    train_dataset = BaseDataImageSet(cfg, mode='train', img_suffix=cfg.DATASETS.IMG_SUFFIX, seg_map_suffix=cfg.DATASETS.SEG_MAP_SUFFIX,  main_transform=train_main_transform, reduce_zero_label= cfg.DATASETS.REDUCE_ZERO_LABEL
                                     )
    val_dataset = BaseDataImageSet(cfg, mode='val', img_suffix=cfg.DATASETS.IMG_SUFFIX, seg_map_suffix=cfg.DATASETS.SEG_MAP_SUFFIX,  main_transform=val_main_transform, reduce_zero_label= cfg.DATASETS.REDUCE_ZERO_LABEL
                                   )

    train_loader = DataLoaderX(
        train_dataset, batch_size=cfg.SOLVER.PER_BATCH, shuffle=True,
        num_workers=num_workers, pin_memory=True,   drop_last=cfg.DATALOADER.DROP_LAST
    )
    val_loader = DataLoaderX(
        val_dataset, batch_size=cfg.SOLVER.PER_BATCH, shuffle=False,
        num_workers=num_workers
    )
    return train_loader, val_loader
