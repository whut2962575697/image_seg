import os
from skimage.io import imread
from prettytable import PrettyTable
from tqdm import tqdm
import numpy as np
import shutil

def print_dataset_statistics(img_path, seg_map_path, class_names=[], img_suffix='.png', seg_map_suffix='.png', datatype='dataset'):
    R_means = []
    G_means = []
    B_means = []
    R_stds = []
    G_stds = []
    B_stds = []
    filenames = [x for x in os.listdir(img_path) if x.endswith(img_suffix)]
    pix_nums = [0]*len(class_names)
    for filename in tqdm(filenames):
        img = imread(os.path.join(img_path, filename))
        im_R = img[:, :, 0] / 255.0
        im_G = img[:, :, 1] / 255.0
        im_B = img[:, :, 2] / 255.0
        im_R_mean = np.mean(im_R)
        im_G_mean = np.mean(im_G)
        im_B_mean = np.mean(im_B)
        im_R_std = np.std(im_R)
        im_G_std = np.std(im_G)
        im_B_std = np.std(im_B)
        R_means.append(im_R_mean)
        G_means.append(im_G_mean)
        B_means.append(im_B_mean)
        R_stds.append(im_R_std)
        G_stds.append(im_G_std)
        B_stds.append(im_B_std)
        gt = imread(os.path.join(seg_map_path, filename.replace(img_suffix, seg_map_suffix)))
        for _i in range(len(pix_nums)):
            pix_nums[_i] = pix_nums[_i] + sum(gt.flatten()==_i)
        
    a = [R_means, G_means, B_means]
    b = [R_stds, G_stds, B_stds]
    mean = [0, 0, 0]
    std = [0, 0, 0]
    mean[0] = np.mean(a[0])
    mean[1] = np.mean(a[1])
    mean[2] = np.mean(a[2])
    std[0] = np.mean(b[0])
    std[1] = np.mean(b[1])
    std[2] = np.mean(b[2])
    print('数据集{}的RGB平均值为\n[{},{},{}]'.format(datatype,mean[0], mean[1], mean[2]))
    print('数据集{}的RGB方差为\n[{},{},{}]'.format(datatype,std[0], std[1], std[2]))
    table = PrettyTable(["序号", "名称", "pix_num"])
    for i in range(len(pix_nums)):
        table.add_row([i, class_names[i], pix_nums[i]])
    print(table)
        
        
if __name__ == "__main__":

    for img_filename in os.listdir(r'/media/dell/E2DE40E3DE40B219/tc/suichang_round1_train_210120'):
        if img_filename.endswith('.png'):
            shutil.copy(os.path.join(r'/media/dell/E2DE40E3DE40B219/tc/suichang_round1_train_210120', img_filename),
                        os.path.join(r'/media/dell/E2DE40E3DE40B219/tc/all_data/gts', img_filename))

    for img_filename in os.listdir(r'/media/dell/E2DE40E3DE40B219/tc/suichang_round1_train_210120'):
        if img_filename.endswith('.tif'):
            shutil.copy(os.path.join(r'/media/dell/E2DE40E3DE40B219/tc/suichang_round1_train_210120', img_filename),
                        os.path.join(r'/media/dell/E2DE40E3DE40B219/tc/all_data/imgs', img_filename))

    for img_filename in os.listdir(r'/media/dell/E2DE40E3DE40B219/tc/suichang_round2_train_210316'):
        if img_filename.endswith('.png'):
            shutil.copy(os.path.join(r'/media/dell/E2DE40E3DE40B219/tc/suichang_round2_train_210316', img_filename),
                        os.path.join(r'/media/dell/E2DE40E3DE40B219/tc/all_data/gts', img_filename))

    for img_filename in os.listdir(r'/media/dell/E2DE40E3DE40B219/tc/suichang_round2_train_210316'):
        if img_filename.endswith('.tif'):
            shutil.copy(os.path.join(r'/media/dell/E2DE40E3DE40B219/tc/suichang_round2_train_210316', img_filename),
                        os.path.join(r'/media/dell/E2DE40E3DE40B219/tc/all_data/imgs', img_filename))

    # print_dataset_statistics(r'/cache/hw_rs/imgs/train', r'/cache/hw_rs/gts/train', class_names=['道路', '背景'])
        
    