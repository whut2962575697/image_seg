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
from skimage.color import gray2rgb
import os.path as osp
import numpy as np
import cv2
import albumentations as A


class BaseDataImageSet(data.Dataset):
    """
    Base dataset for semantic segmentation(train|val).
    Args:
        cfg(yacs.config): Config to the pipeline.
        mode(str): Mode of the pipeline. "train|val".
        img_suffix(str): Suffix of images. ".png|.tif|.jpg".
        seg_map_suffix(str): Suffix of segmentation maps. ".png".
        main_transform(albumentations): Data transform for semantic segmentation.
        reduce_zero_label(bool): Whether to mark label zero as ignored. True|False

    """
    def __init__(self, cfg, mode, img_suffix, seg_map_suffix, main_transform, reduce_zero_label=False):
        self.mode = mode
        self.cfg = cfg
        self.img_suffix = img_suffix
        self.seg_map_suffix = seg_map_suffix
        self.main_transform = main_transform
        self.file_client = None
        self.reduce_zero_label = reduce_zero_label

        self.mode = mode
        if self.mode == 'train':
            self.file_list = [x for x in
                              os.listdir(os.path.join(cfg.DATASETS.DATA_PATH, cfg.DATASETS.TRAIN_IMAGE_FOLDER))
                              if x.endswith(img_suffix)]
        elif self.mode == 'val':
            self.file_list = [x for x in
                              os.listdir(os.path.join(cfg.DATASETS.DATA_PATH, cfg.DATASETS.VAL_IMAGE_FOLDER))
                              if x.endswith(img_suffix)]

        self.num_samples = len(self.file_list)

    def read_image(self, img_path, color_type=None):
        if color_type == 'color':
            im = imread(img_path)
            if im.ndim < 3:
                im = gray2rgb(im)
        else:
            im = imread(img_path)

        return im

    def __getitem__(self, index):
        if not hasattr(self, 'post_transforms'):
            self._split_transforms()
        # read data and main transforms
        img, gt = self.read_data_and_gt(index)

        if self.cfg.INPUT.USE_COPY_PASTE and self.mode == 'train':
            # if use copy_paste should split transforms
            masks = []
            bboxes = []
            # split the mask into individual binary masks for each class
            for ix, value in enumerate(np.unique(gt)):
                if value == 255:
                    continue
                masks.append(gt == value)
                bboxes.append(self.extract_bbox(gt == value) + (value, ix))
            paste_idx = random.randint(0, self.__len__() - 1)
            paste_img, paste_gt = self.read_data_and_gt(paste_idx)

            paste_masks = []
            paste_bboxes = []
            # split the mask into individual binary masks for each class

            for ix, value in enumerate(np.unique(paste_gt)):
                if value == 255:
                    continue
                paste_masks.append(paste_gt == value)
                paste_bboxes.append(self.extract_bbox(paste_gt == value) + (value, ix))
            data = {'image': img, 'masks': masks, 'bboxes': bboxes}
            paste_data = {'paste_masks': paste_masks, 'paste_bboxes':paste_bboxes, 'paste_image':paste_img}

            aug = self.copy_paste(**data, **paste_data)
            img, masks, bboxes = aug['image'], aug['masks'], aug['bboxes']

            mask_classes = [b[-2] for b in bboxes]
            mask_indices = [b[-1] for b in bboxes]

            semantic_mask = np.zeros_like(masks[0]).astype(np.long)  # could be uint8 if fewer than 255 classes
            for _class, index in zip(mask_classes, mask_indices):
                semantic_mask += masks[index] * _class
            gt = semantic_mask

            data = {'image': img, 'mask': gt}
            aug = self.post_transforms(**data)
            img, gt = aug['image'], aug['mask']

        if self.reduce_zero_label:
            gt[gt == 0] = 255
            gt = gt - 1
            gt[gt == 254] = 255

        return img, gt.long()

    def __len__(self):
        return self.num_samples

    def _split_transforms(self):
        split_index = None
        for ix, tf in enumerate(list(self.main_transform.transforms)):
            if tf.get_class_fullname() == 'copypaste.CopyPaste':
                split_index = ix

        if split_index is not None:
            tfs = list(self.main_transform.transforms)
            pre_copy = tfs[:split_index]
            copy_paste = tfs[split_index]
            post_copy = tfs[split_index + 1:]

            # replicate the other augmentation parameters
            bbox_params = None
            keypoint_params = None
            paste_additional_targets = {}
            if 'bboxes' in self.main_transform.processors:
                bbox_params = self.main_transform.processors['bboxes'].params
                paste_additional_targets['paste_bboxes'] = 'bboxes'
                if self.main_transform.processors['bboxes'].params.label_fields:
                    msg = "Copy-paste does not support bbox label_fields! "
                    msg += "Expected bbox format is (a, b, c, d, label_field)"
                    raise Exception(msg)
            if 'keypoints' in self.main_transform.processors:
                keypoint_params = self.main_transform.processors['keypoints'].params
                paste_additional_targets['paste_keypoints'] = 'keypoints'
                if keypoint_params.label_fields:
                    raise Exception('Copy-paste does not support keypoint label fields!')

            if self.main_transform.additional_targets:
                raise Exception('Copy-paste does not support additional_targets!')

            # recreate transforms
            self.main_transform = A.Compose(pre_copy, bbox_params, keypoint_params, additional_targets=None)
            self.post_transforms = A.Compose(post_copy, bbox_params, keypoint_params, additional_targets=None)
            self.copy_paste = A.Compose(
                [copy_paste], bbox_params, keypoint_params, additional_targets=paste_additional_targets
            )
        else:
            self.copy_paste = None
            self.post_transforms = None

    def extract_bbox(self, mask):
        h, w = mask.shape
        yindices = np.where(np.any(mask, axis=0))[0]
        xindices = np.where(np.any(mask, axis=1))[0]
        if yindices.shape[0]:
            y1, y2 = yindices[[0, -1]]
            x1, x2 = xindices[[0, -1]]
            y2 += 1
            x2 += 1
        else:
            y1, x1, y2, x2 = 0, 0, 0, 0

        return (y1, x1, y2, x2)

    def read_data_and_gt(self, index):
        if self.mode == 'train':
            img = self.read_image(os.path.join(self.cfg.DATASETS.DATA_PATH, self.cfg.DATASETS.TRAIN_IMAGE_FOLDER, self.file_list[index]), color_type='color')
            gt = self.read_image(
                os.path.join(self.cfg.DATASETS.DATA_PATH, self.cfg.DATASETS.TRAIN_GT_FOLDER,
                             self.file_list[index].replace(self.img_suffix, self.seg_map_suffix)),
                color_type='unchanged')
        else:
            img = self.read_image(os.path.join(self.cfg.DATASETS.DATA_PATH, self.cfg.DATASETS.VAL_IMAGE_FOLDER, self.file_list[index]), color_type='color')
            gt = self.read_image(
                os.path.join(self.cfg.DATASETS.DATA_PATH, self.cfg.DATASETS.VAL_GT_FOLDER,
                             self.file_list[index].replace(self.img_suffix, self.seg_map_suffix)),
                color_type='unchanged')

        img = img[:, :, :self.cfg.MODEL.N_CHANNEL]
        # base transform
        data = {'image': img, 'mask': gt}
        aug = self.main_transform(**data)
        img, gt = aug['image'], aug['mask']
        return img, gt



    def get_num_samples(self):
        return self.num_samples


class BaseTestDataImageSet(data.Dataset):
    """
        Base dataset for semantic segmentation(test).
        Args:
            cfg(yacs.config): Config to the pipeline.
            img_suffix(str): Suffix of images. ".png|.tif|.jpg".
            main_transform(albumentations): Data transform for semantic segmentation.

        """
    def __init__(self, cfg, img_suffix, main_transform):
        self.cfg = cfg
        self.img_suffix = img_suffix
        self.main_transform = main_transform

        self.file_list = [x for x in os.listdir(cfg.TEST.IMAGE_FOLDER) if x.endswith(img_suffix)]

        self.num_samples = len(self.file_list)

    def read_image(self, img_path, color_type=None):
        if color_type == 'color':
            im = imread(img_path)
            if im.ndim < 3:
                im = gray2rgb(im)
        else:
            im = imread(img_path)

        return im

    def __getitem__(self, index):
        data, filename = self.read_data(index)
        return data, filename

    def __len__(self):
        return self.num_samples

    def read_data(self, index):
       
        img = self.read_image(os.path.join(self.cfg.TEST.IMAGE_FOLDER, self.file_list[index]))
        img = img[:, :, :self.cfg.MODEL.N_CHANNEL]
        data = {'image': img}
        aug = self.main_transform(**data)
        img = aug['image']
        return img, self.file_list[index]

    def get_num_samples(self):
        return self.num_samples





