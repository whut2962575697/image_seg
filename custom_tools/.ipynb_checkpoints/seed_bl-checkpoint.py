# -*- encoding: utf-8 -*-
'''
@File    :   seed_bl.py    
@Contact :   whut.hexin@foxmail.com
@License :   (C)Copyright 2017-2020, HeXin

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2020/9/20 10:36   xin      1.0         None
'''


from skimage.io import imread, imsave
import os
import numpy as np
import shutil
from tqdm import tqdm
import random
import cv2

from PIL import Image
Image.MAX_IMAGE_PIXELS = None

"""
seed 病理图像识别赛道数据预处理脚本
"""


#  阴性样本
def prepare1(root, save_path, save_img_path):
    imgs = os.listdir(root)

    for img_filename in tqdm(imgs):
        shutil.copy(os.path.join(root, img_filename), os.path.join(save_img_path, img_filename))
        img = imread(os.path.join(root, img_filename))
        gt = np.zeros(img.shape, np.uint8)
        # imsave(os.path.join(save_path, img_filename), gt)
        cv2.imwrite(os.path.join(save_path, img_filename.replace('_mask', '').replace('jpg', 'png')), gt)


# 阳性样本
def prepare2(root, save_path, save_img_path):
    imgs = os.listdir(root)
    for img_filename in tqdm(imgs):
        if 'mask' in img_filename:
            img = imread(os.path.join(root, img_filename))

            # gt = np.zeros(img.shape, np.uint8)
            # for i, row in enumerate(img):
            #     for j, cell in enumerate(row):
            #         if cell > 100:
            #             gt[i, j] = 1
            # imsave(os.path.join(save_path, img_filename), gt)
            img[img < 100] = 0
            img[img>100] = 1
            # imsave(os.path.join(save_path, img_filename), img)
            cv2.imwrite(os.path.join(save_path, img_filename.replace('_mask', '').replace('jpg', 'png')), img)

        else:
            shutil.copy(os.path.join(root, img_filename), os.path.join(save_img_path, img_filename))


def split_dataset(img_path, gt_path, split_rate, save_path):
    os.mkdir(os.path.join(save_path, 'train'))
    os.mkdir(os.path.join(save_path, 'train', 'img'))
    os.mkdir(os.path.join(save_path, 'train', 'gt'))

    os.mkdir(os.path.join(save_path, 'val'))
    os.mkdir(os.path.join(save_path, 'val', 'img'))
    os.mkdir(os.path.join(save_path, 'val', 'gt'))

    img_filenames = os.listdir(img_path)
    random.shuffle(img_filenames)
    assert split_rate>0 and split_rate<1
    for img_filename in tqdm(img_filenames[:int(split_rate*len(img_filenames))]):
        shutil.copy(os.path.join(img_path, img_filename), os.path.join(save_path, 'train', 'img', img_filename))
        shutil.copy(os.path.join(gt_path, img_filename.replace('jpg', 'png')), os.path.join(save_path, 'train', 'gt', img_filename.replace('jpg', 'png')))
    for img_filename in tqdm(img_filenames[int(split_rate*len(img_filenames)):]):
        shutil.copy(os.path.join(img_path, img_filename), os.path.join(save_path, 'val', 'img', img_filename))
        shutil.copy(os.path.join(gt_path, img_filename.replace('jpg', 'png')), os.path.join(save_path, 'val', 'gt', img_filename.replace('jpg', 'png')))







if __name__ == "__main__":
    split_dataset(r'D:\xf\seed_bl\imgs', r'D:\xf\seed_bl\gts', 0.8, r'D:\xf\seed_bl\dataset')
    # prepare1(r'D:\xf\医疗卫生初赛数据集01\医疗卫生初赛数据集01', r'D:\xf\seed_bl\gts', r'D:\xf\seed_bl\imgs')
    # prepare1(r'D:\xf\医疗卫生初赛数据集03\医疗卫生初赛数据集03', r'D:\xf\seed_bl\gts', r'D:\xf\seed_bl\imgs')
    # prepare2(r'D:\xf\医疗卫生初赛数据集02\医疗卫生初赛数据集02', r'D:\xf\seed_bl\gts', r'D:\xf\seed_bl\imgs')
    # prepare2(r'D:\xf\医疗卫生初赛数据集04\医疗卫生初赛数据集04', r'D:\xf\seed_bl\gts', r'D:\xf\seed_bl\imgs')