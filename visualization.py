from skimage.io import imread, imsave
import numpy as np
import os
from tqdm import tqdm
import shutil
import json

color_key_list = [
    {'color':(0, 200, 0), 'label': '林地', 'cls_num': 1, 'cls_name': 'forest'},
    {'color':(150, 250, 0), 'label': '草地', 'cls_num': 2, 'cls_name': 'grass'},
    {'color':(150, 200, 150), 'label': '耕地', 'cls_num': 3, 'cls_name': 'farmland'},
    {'color':(200, 0, 200), 'label': '水域', 'cls_num': 4, 'cls_name': 'water'},
    {'color':(150, 0, 250), 'label': '道路', 'cls_num': 5, 'cls_name': 'road'},
    {'color':(150, 150, 250), 'label': '城镇建设用地', 'cls_num': 6, 'cls_name': 'urban_area'},
    {'color':(250, 200, 0), 'label': '农村建设用地', 'cls_num': 7, 'cls_name': 'countryside'},
    {'color':(25, 25, 112), 'label': '工业用地', 'cls_num': 8, 'cls_name': 'industrial_land'},
    {'color':(255, 0, 0), 'label': '构筑物', 'cls_num': 9, 'cls_name': 'construction'},
    {'color':(255, 20, 147), 'label': '裸地', 'cls_num': 10, 'cls_name': 'bareland'},
    {'color':(255, 228, 181), 'label': '密林', 'cls_num': 11, 'cls_name': 'denseForest'},
    {'color':(0, 134, 139), 'label': '疏林', 'cls_num': 12, 'cls_name': 'openForest'},
    {'color':(0, 255, 255), 'label': 'others', 'cls_num': 13, 'cls_name': 'others'},
                 ]


def re_color(gray_img_path, save_path, reduce_zero_label=False):
    filenames = os.listdir(gray_img_path)

    for filename in tqdm(filenames):

        gray_img = imread(os.path.join(gray_img_path, filename))
        color_img = np.ones((gray_img.shape[0], gray_img.shape[1], 3)).astype(np.uint8)

        for i, color in enumerate(color_key_list):
            if reduce_zero_label:
                i = i + 1
            color_img[gray_img==i] = color['color']
        imsave(os.path.join(save_path, filename), color_img)
        # for i, row in enumerate(gray_img):
        #     for j, cell in enumerate(row):
        #         if reduce_zero_label:
        #             cell = cell - 1
        #         color_img[i, j] = color_key_dic[cell]['color']


if __name__ == "__main__":
    import argparse


    parser = argparse.ArgumentParser(description="Baseline Training")
    parser.add_argument("--gray_path", default="test", help="gray_path", type=str)
    parser.add_argument("--color_path", default="test", help="output path", type=str)
    parser.add_argument("--reduce_zero_label", default=False, help="reduce_zero_label", type=bool)

    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    re_color(args.gray_path, args.color_path, args.reduce_zero_label)



