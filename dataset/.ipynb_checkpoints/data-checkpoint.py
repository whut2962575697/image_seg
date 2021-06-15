# -*- encoding: utf-8 -*-
'''
@File    :   data.py    
@Contact :   whut.hexin@foxmail.com
@License :   (C)Copyright 2017-2018, HeXin

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2020/1/28 14:11   xin      1.0         None
'''

from torch.utils import data
import os
import torch
import random
from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True
from skimage.io import imread
import os.path as osp
import numpy as np
import cv2
# import mmcv

from utils.augmix.augmix import augment_and_mix



class BaseDataImageSet(data.Dataset):
    def __init__(self, cfg, mode, img_suffix, seg_map_suffix, main_transform):
        self.mode = mode
        self.cfg = cfg
        self.img_suffix = img_suffix
        self.seg_map_suffix = seg_map_suffix
        self.main_transform = main_transform
        self.file_client = None

        self.mode = mode
        if self.mode == 'train':
            self.file_list = [x for x in
                              os.listdir(os.path.join(cfg.DATASETS.DATA_PATH, cfg.DATASETS.IMAGE_FOLDER, 'train'))
                              if x.endswith(img_suffix)]
        elif self.mode == 'val':
            self.file_list = [x for x in
                              os.listdir(os.path.join(cfg.DATASETS.DATA_PATH, cfg.DATASETS.IMAGE_FOLDER, 'val'))
                              if x.endswith(img_suffix)]
        

        self.num_samples = len(self.file_list)


    def read_image(self, img_path, color_type='color', imdecode_backend ='cv2'):
        # if self.file_client is None:
        #     self.file_client = mmcv.FileClient(**dict(backend='disk'))
        # img_bytes = self.file_client.get(img_path)
        # im= mmcv.imfrombytes(
        #     img_bytes, flag=color_type, backend=imdecode_backend)
        # # im = mmcv.imread(img_path, flag='unchanged', backend=imdecode_backend)
        im = imread(img_path)
        return im

    def __getitem__(self, index):
        data, gt = self.read_data_and_gt(index)
        return data, gt
          

    def __len__(self):
        return self.num_samples

    def read_data_and_gt(self, index):
        if self.mode == 'train':
            img = self.read_image(os.path.join(self.cfg.DATASETS.DATA_PATH, self.cfg.DATASETS.IMAGE_FOLDER, 'train', self.file_list[index]))
        elif self.mode == 'val':
            img = self.read_image(os.path.join(self.cfg.DATASETS.DATA_PATH, self.cfg.DATASETS.IMAGE_FOLDER, 'val', self.file_list[index]))

        if self.mode == 'train':
            gt = self.read_image(
                os.path.join(self.cfg.DATASETS.DATA_PATH, self.cfg.DATASETS.GT_FOLDER, 'train', self.file_list[index].replace(self.img_suffix, self.seg_map_suffix)),
                color_type='unchanged', imdecode_backend='pillow')
        elif self.mode == 'val':
            gt = self.read_image(
                os.path.join(self.cfg.DATASETS.DATA_PATH, self.cfg.DATASETS.GT_FOLDER, 'val', self.file_list[index].replace(self.img_suffix, self.seg_map_suffix)),
                color_type='unchanged', imdecode_backend='pillow')
        # print(img.shape, gt.shape)
        img = img[:, :, :3]
        gt = np.array(gt)[:224,:224]
#         if self.mode == 'val':
#             print(gt)
        gt[gt <=0] = 1
        gt[gt>12] = 1
        gt = gt - 1
#         if self.mode == 'val':
#             print(gt)

        data = {'image': img, 'mask':gt}
        aug = self.main_transform(**data)
        img, gt = aug['image'], aug['mask']
        # if self.cfg.INPUT.USE_AUGMIX and self.mode == 'train':
        #     img = augment_and_mix(Image.fromarray(img), self.image_transform)
        # else:
        #     img = self.image_transform(img)
        # gt = self.main_transform(gt)
        # gt = torch.LongTensor(np.array(gt))
        # print(gt)

        return img, gt.long()

    def get_num_samples(self):
        return self.num_samples





