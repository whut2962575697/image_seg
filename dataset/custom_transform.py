import math
import numbers
# import random
import cv2

from PIL import Image, ImageOps
import numpy as np
from numpy import random


def objs_from_gt_array(gt_array, max_obj_per_class=30):
    '''
    提取栅格图像中的对象，先标记不同序号（序号不顺序），然后按x、y顺序标记重新提取对象
    输入的灰度矩阵要求：
    1.输入灰度gt图像
    2.类别数小于max_obj_per_class
    3.单个样本中，每个类的对象数小于等于10
    :param gt_array: 灰度gt矩阵
    :return:
    '''

    def num_swift(matrix, add):
        '''
        提取对象辅助函数，对矩阵中非0数做加法运算
        :param matrix: 矩阵
        :param add: 加数
        :return:
        '''
        matrix[matrix > 0] += add
        return matrix

    objs_temp = np.zeros_like(gt_array, dtype=np.uint8)
    gt_clss = split_by_class(gt_array)
    for index, gt_cls in enumerate(gt_clss):
        _, labels = cv2.connectedComponents(gt_cls, connectivity=8)
        objs_temp += np.array(num_swift(labels, index * max_obj_per_class), dtype=np.uint8)
    obj_split = split_by_object(objs_temp)
    return obj_split

# endregion


# region 提取对象辅助函数：按类分离gt、按顺序重新分离对象

def split_by_class(gt_array):
    '''
    按类别将gt提取出多个gt_cls，对应的各类为1、背景为0，共class_num个gt_cls
    :param gt_array: 灰度gt矩阵
    :return:
    '''
    gt_clss = []
    clss = np.unique(gt_array)
    for cls in clss:
        gt_cls = np.zeros_like(gt_array)
        gt_cls[gt_array == cls] = 1
        gt_clss.append(gt_cls)
    return gt_clss


def split_by_object(objs_temp):
    '''
    按横纵顺序，将无序obj array提取各个顺序obj
    :param objs_temp: 无序obj array
    :return:
    '''
    cls_objs = []
    max_num, idx = np.unique(objs_temp, return_index=True)
    max_num = objs_temp.ravel()[np.sort(idx)]
    for index, num in enumerate(max_num):
        cls_obj = np.zeros_like(objs_temp)
        cls_obj[objs_temp == num] = 1
        cls_objs.append(cls_obj)
    return cls_objs

# endregion




class PhotoMetricDistortion(object):
    """Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last.
    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)
    8. randomly swap channels
    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
    """

    def __init__(self,
                 brightness_delta=32,
                 contrast_range=(0.5, 1.5),
                 saturation_range=(0.5, 1.5),
                 hue_delta=18):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def convert(self, img, alpha=1, beta=0):
        """Multiple with alpha and add beat with clip."""
        img = img.astype(np.float32) * alpha + beta
        img = np.clip(img, 0, 255)
        return img.astype(np.uint8)

    def brightness(self, img):
        """Brightness distortion."""
        if random.randint(2):
            return self.convert(
                img,
                beta=random.uniform(-self.brightness_delta,
                                    self.brightness_delta))
        return img

    def contrast(self, img):
        """Contrast distortion."""
        if random.randint(2):
            return self.convert(
                img,
                alpha=random.uniform(self.contrast_lower, self.contrast_upper))
        return img

    def saturation(self, img):
        """Saturation distortion."""
        if random.randint(2):
            # img = mmcv.bgr2hsv(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            img[:, :, 1] = self.convert(
                img[:, :, 1],
                alpha=random.uniform(self.saturation_lower,
                                     self.saturation_upper))
            # img = mmcv.hsv2bgr(img)
            img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        return img

    def hue(self, img):
        """Hue distortion."""
        if random.randint(2):
            # img = mmcv.bgr2hsv(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            img[:, :,
                0] = (img[:, :, 0].astype(int) +
                      random.randint(-self.hue_delta, self.hue_delta)) % 180
            # img = mmcv.hsv2bgr(img)
            img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        return img

    def __call__(self, img):
        """Call function to perform photometric distortion on images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Result dict with images distorted.
        """

        # img = results['img']
        # random brightness
        img = self.brightness(img)

        # mode == 0 --> do random contrast first
        # mode == 1 --> do random contrast last
        mode = random.randint(2)
        if mode == 1:
            img = self.contrast(img)

        # random saturation
        img = self.saturation(img)

        # random hue
        img = self.hue(img)

        # random contrast
        if mode == 0:
            img = self.contrast(img)

        # results['img'] = img
        return img

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(brightness_delta={self.brightness_delta}, '
                     f'contrast_range=({self.contrast_lower}, '
                     f'{self.contrast_upper}), '
                     f'saturation_range=({self.saturation_lower}, '
                     f'{self.saturation_upper}), '
                     f'hue_delta={self.hue_delta})')
        return repr_str
