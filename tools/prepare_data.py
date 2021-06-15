# # -*- encoding: utf-8 -*-
# '''
# @File    :   prepare_data.py
# @Contact :   whut.hexin@foxmail.com
# @License :   (C)Copyright 2017-2018, HeXin
#
# @Modify Time      @Author    @Version    @Desciption
# ------------      -------    --------    -----------
# 2020/1/28 13:20   xin      1.0         None
# '''
# from skimage.io import imread, imsave
# import numpy as np
# import os
# from tqdm import tqdm
# import cv2
#
# # color_2_cls = {
# #     (0, 0, 0): 0,
# #     (128, 0, 0): 1,
# #     (75, 0, 130): 2,
# #     (255, 215, 0): 3,
# #     (0, 0, 128): 4,
# #     (128, 128, 128): 5,
# #     (0, 128, 128): 6,
# #     (72, 209, 204): 7,
# #     (255, 0, 0): 8,
# #     (34, 139, 34): 9,
# #     (255, 0, 255): 10,
# # }
#
# color_2_cls = {
#     (0,200,0): 0,
#     (150,250,0): 1,
#     (150,200,150): 2,
#     (200,0,200): 3,
#     (150,0,250): 4,
#     (150,150,250): 5,
#     (250,200,0): 6,
#     (200,200,0): 7,
#     (200,0,0): 8,
#     (250,0,150): 9,
#     (200,150,150): 10,
#     (250,150,150): 11,
#     (0,0,200): 12,
#     (0,150,200): 13,
#     (0,200,250): 14,
#     (0,0,0): 15
# }
#
#
# def color_to_gray(color_img_file, save_file, bg_value=(0, 0, 0)):
#     color_img = imread(color_img_file)
#     gray_img = np.zeros((color_img.shape[0], color_img.shape[1]))
#     for i, row in enumerate(color_img):
#         for j, cell in enumerate(row):
#             if tuple(cell) in color_2_cls:
#                 gray_img[i, j] = color_2_cls[tuple(cell)]
#             else:
#                  gray_img[i, j] = color_2_cls[bg_value]
#     imsave(save_file, gray_img)
#
#
# def calculate_colormap(color_img_file, gray_img_file):
#     color_img = imread(color_img_file)
#     gray_img = imread(gray_img_file)
#     color_map = {}
#
#     for color_row, gray_row in zip(color_img, gray_img):
#         for color_cell, gray_cell in zip(color_row, gray_row):
#             if tuple(color_cell) not in color_map:
#                 color_map[tuple(color_cell)] = gray_cell
#
#     print(color_map)
#
#
# def split_test_area(rs_file, gray_gt_file, color_gt_file, save_path):
#     rs_img = imread(rs_file)
#     gray_gt_img = imread(gray_gt_file)
#     color_gt_img = imread(color_gt_file)
#     [h, w] = rs_img.shape[:2]
#     print('##### img height: {0}, img width: {1}'.format(h, w))
#     test_area_rs_img = rs_img[int(0.5*h):, int(0.5*w):]
#     test_area_gray_gt_img = gray_gt_img[int(0.5 * h):, int(0.5 * w):]
#     test_area_color_gt_img = color_gt_img[int(0.5 * h):, int(0.5 * w):]
#
#     train_area1_rs_img = rs_img[:int(0.5*h), :int(0.5*w)]
#     train_area1_gray_gt_img = gray_gt_img[:int(0.5 * h), :int(0.5 * w)]
#     train_area1_color_gt_img = color_gt_img[:int(0.5 * h), :int(0.5 * w)]
#
#     train_area2_rs_img = rs_img[:int(0.5 * h), int(0.5 * w):]
#     train_area2_gray_gt_img = gray_gt_img[:int(0.5 * h), int(0.5 * w):]
#     train_area2_color_gt_img = color_gt_img[:int(0.5 * h), int(0.5 * w):]
#
#     train_area3_rs_img = rs_img[int(0.5 * h):, :int(0.5 * w)]
#     train_area3_gray_gt_img = gray_gt_img[int(0.5 * h):, :int(0.5 * w)]
#     train_area3_color_gt_img = color_gt_img[int(0.5 * h):, :int(0.5 * w)]
#
#     # test area
#     print('#################split test area###################')
#     imsave(os.path.join(save_path, 'test_rs.tif'), test_area_rs_img)
#     imsave(os.path.join(save_path, 'test_gray_gt.tif'), test_area_gray_gt_img)
#     imsave(os.path.join(save_path, 'test_color_gt.tif'), test_area_color_gt_img)
#
#     train_rs_img_list = [train_area1_rs_img, train_area2_rs_img, train_area3_rs_img]
#     train_gray_gt_img_list = [train_area1_gray_gt_img, train_area2_gray_gt_img, train_area3_gray_gt_img]
#     train_color_gt_img_list = [train_area1_color_gt_img, train_area2_color_gt_img, train_area3_color_gt_img]
#
#     # trainVal area
#     print('#################split trainVal area###################')
#     for i, train_area_rs_img, train_area_gray_gt_img, train_area_color_gt_img in zip(list(range(3)), train_rs_img_list,
#                                                         train_gray_gt_img_list, train_color_gt_img_list):
#
#         print('#################split train area##################')
#         train_rs = train_area_rs_img[:int(0.8*train_area_rs_img.shape[0]), :]
#         train_gary = train_area_gray_gt_img[:int(0.8*train_area_gray_gt_img.shape[0]), :]
#         train_color = train_area_color_gt_img[:int(0.8 * train_area_color_gt_img.shape[0]), :]
#
#         imsave(os.path.join(save_path, 'train{0}_rs.tif'.format(i)), train_rs)
#         imsave(os.path.join(save_path, 'train{0}_gray_gt.tif'.format(i)), train_gary)
#         imsave(os.path.join(save_path, 'train{0}_color_gt.tif' .format(i)), train_color)
#
#         print('#################split val area##################')
#
#         val_rs = train_area_rs_img[int(0.8 * train_area_rs_img.shape[0]):, :]
#         val_gary = train_area_gray_gt_img[int(0.8 * train_area_gray_gt_img.shape[0]):, :]
#         val_color = train_area_color_gt_img[int(0.8 * train_area_color_gt_img.shape[0]):, :]
#
#         imsave(os.path.join(save_path, 'val{0}_rs.tif'.format(i)), val_rs)
#         imsave(os.path.join(save_path, 'val{0}_gray_gt.tif'.format(i)), val_gary)
#         imsave(os.path.join(save_path, 'val{0}_color_gt.tif'.format(i)), val_color)
#
#     # random cut 5000 image samples for train and val
#     print('#################random crop samples###################')
#     if not os.path.exists(os.path.join(save_path, 'train_rs_imgs')):
#         os.mkdir(os.path.join(save_path, 'train_rs_imgs'))
#     if not os.path.exists(os.path.join(save_path, 'train_gray_gt_imgs')):
#         os.mkdir(os.path.join(save_path, 'train_gray_gt_imgs'))
#     if not os.path.exists(os.path.join(save_path, 'train_color_gt_imgs')):
#         os.mkdir(os.path.join(save_path, 'train_color_gt_imgs'))
#
#     if not os.path.exists(os.path.join(save_path, 'val_rs_imgs')):
#         os.mkdir(os.path.join(save_path, 'val_rs_imgs'))
#     if not os.path.exists(os.path.join(save_path, 'val_gray_gt_imgs')):
#         os.mkdir(os.path.join(save_path, 'val_gray_gt_imgs'))
#     if not os.path.exists(os.path.join(save_path, 'val_color_gt_imgs')):
#         os.mkdir(os.path.join(save_path, 'val_color_gt_imgs'))
#
#     print('#################random crop train samples###################')
#     crop_size = 224
#     for k in tqdm(list(range(4000)), total=len(list(range(4000))),
#                               leave=False):
#         img_index = np.random.randint(0, 3)
#         rs_img = train_rs_img_list[img_index]
#         gray_gt_img = train_gray_gt_img_list[img_index]
#         color_gt_img = train_color_gt_img_list[img_index]
#         rs_img = rs_img[:int(0.8*rs_img.shape[0]), :]
#         gray_gt_img = gray_gt_img[:int(0.8*gray_gt_img.shape[0]), :]
#         color_gt_img = color_gt_img[:int(0.8*color_gt_img.shape[0]), :]
#         random_h_index = np.random.randint(0, rs_img.shape[0]-crop_size-1)
#         random_w_index = np.random.randint(0, rs_img.shape[1] - crop_size-1)
#         crop_rs_img = rs_img[random_h_index:random_h_index+crop_size, random_w_index:random_w_index+crop_size]
#         crop_gray_gt_img = gray_gt_img[random_h_index:random_h_index + crop_size,
#                            random_w_index:random_w_index + crop_size]
#         crop_color_gt_img = color_gt_img[random_h_index:random_h_index + crop_size,
#                            random_w_index:random_w_index + crop_size]
#         imsave(os.path.join(save_path, 'train_rs_imgs', str(k)+'.tif'), crop_rs_img)
#         imsave(os.path.join(save_path, 'train_gray_gt_imgs', str(k) + '.tif'), crop_gray_gt_img)
#         imsave(os.path.join(save_path, 'train_color_gt_imgs', str(k) + '.tif'), crop_color_gt_img)
#
#     print('#################random crop val samples###################')
#     for k in tqdm(list(range(1000)), total=len(list(range(1000))),
#                               leave=False):
#         img_index = np.random.randint(0, 3)
#         rs_img = train_rs_img_list[img_index]
#         gray_gt_img = train_gray_gt_img_list[img_index]
#         color_gt_img = train_color_gt_img_list[img_index]
#         rs_img = rs_img[int(0.8*rs_img.shape[0]):, :]
#         gray_gt_img = gray_gt_img[int(0.8*gray_gt_img.shape[0]):, :]
#         color_gt_img = color_gt_img[int(0.8*color_gt_img.shape[0]):, :]
#         random_h_index = np.random.randint(0, rs_img.shape[0]-crop_size-1)
#         random_w_index = np.random.randint(0, rs_img.shape[1] - crop_size-1)
#         crop_rs_img = rs_img[random_h_index:random_h_index+crop_size, random_w_index:random_w_index+crop_size]
#         crop_gray_gt_img = gray_gt_img[random_h_index:random_h_index + crop_size,
#                            random_w_index:random_w_index + crop_size]
#         crop_color_gt_img = color_gt_img[random_h_index:random_h_index + crop_size,
#                            random_w_index:random_w_index + crop_size]
#         imsave(os.path.join(save_path, 'val_rs_imgs', 'v'+str(k)+'.tif'), crop_rs_img)
#         imsave(os.path.join(save_path, 'val_gray_gt_imgs', 'v'+str(k) + '.tif'), crop_gray_gt_img)
#         imsave(os.path.join(save_path, 'val_color_gt_imgs', 'v'+str(k) + '.tif'), crop_color_gt_img)
#
#
# def split_test_area_new(rs_file, gray_gt_file, color_gt_file, save_path):
#     np.random.seed(2020)
#     rs_img = imread(rs_file)
#     gray_gt_img = imread(gray_gt_file)
#     color_gt_img = imread(color_gt_file)
#     [h, w] = rs_img.shape[:2]
#     print('##### img height: {0}, img width: {1}'.format(h, w))
#     h_delt = h // 20
#     w_delt = w // 10
#
#     k = 0
#     train_list = list()
#     val_list = list()
#     test_list = list()
#
#     for i in range(20):
#         for j in range(10):
#             if i == 19 and j == 9:
#                 cut_rs = rs_img[i * h_delt:, j * w_delt:]
#                 cut_gray = gray_gt_img[i * h_delt:, j * w_delt:]
#                 cut_color = color_gt_img[i * h_delt:, j * w_delt:]
#             elif i == 19:
#                 cut_rs = rs_img[i * h_delt:, j * w_delt:(j + 1) * w_delt]
#                 cut_gray = gray_gt_img[i * h_delt:, j * w_delt:(j + 1) * w_delt]
#                 cut_color = color_gt_img[i * h_delt:, j * w_delt:(j + 1) * w_delt]
#
#             elif j == 9:
#                 cut_rs = rs_img[i * h_delt:(i + 1) * h_delt, j * w_delt:]
#                 cut_gray = gray_gt_img[i * h_delt:(i + 1) * h_delt, j * w_delt:]
#                 cut_color = color_gt_img[i * h_delt:(i + 1) * h_delt, j * w_delt:]
#             else:
#                 cut_rs = rs_img[i * h_delt:(i + 1) * h_delt, j * w_delt:(j + 1) * w_delt]
#                 cut_gray = gray_gt_img[i * h_delt:(i + 1) * h_delt, j * w_delt:(j + 1) * w_delt]
#                 cut_color = color_gt_img[i * h_delt:(i + 1) * h_delt, j * w_delt:(j + 1) * w_delt]
#             p = np.random.uniform(0, 1)
#             if p > 0.8:
#                 # test area
#                 print('#################split test area###################')
#                 imsave(os.path.join(save_path, 'test_{0}_rs.tif').format(k), cut_rs)
#                 imsave(os.path.join(save_path, 'test_{0}_gray_gt.tif').format(k), cut_gray)
#                 imsave(os.path.join(save_path, 'test_{0}_color_gt.tif').format(k), cut_color)
#                 test_list.append([cut_rs, cut_gray, cut_color])
#             elif p > 0.64:
#                 # val area
#                 print('#################split val area###################')
#                 imsave(os.path.join(save_path, 'val_{0}_rs.tif').format(k), cut_rs)
#                 imsave(os.path.join(save_path, 'val_{0}_gray_gt.tif').format(k), cut_gray)
#                 imsave(os.path.join(save_path, 'val_{0}_color_gt.tif').format(k), cut_color)
#                 val_list.append([cut_rs, cut_gray, cut_color])
#             else:
#                 # train area
#                 print('#################split train area###################')
#                 imsave(os.path.join(save_path, 'train_{0}_rs.tif').format(k), cut_rs)
#                 imsave(os.path.join(save_path, 'train_{0}_gray_gt.tif').format(k), cut_gray)
#                 imsave(os.path.join(save_path, 'train_{0}_color_gt.tif').format(k), cut_color)
#                 train_list.append([cut_rs, cut_gray, cut_color])
#             k = k + 1
#
#     # random cut 5000 image samples for train and val
#     print('#################random crop samples###################')
#     if not os.path.exists(os.path.join(save_path, 'train_rs_imgs')):
#         os.mkdir(os.path.join(save_path, 'train_rs_imgs'))
#     if not os.path.exists(os.path.join(save_path, 'train_gray_gt_imgs')):
#         os.mkdir(os.path.join(save_path, 'train_gray_gt_imgs'))
#     if not os.path.exists(os.path.join(save_path, 'train_color_gt_imgs')):
#         os.mkdir(os.path.join(save_path, 'train_color_gt_imgs'))
#
#     if not os.path.exists(os.path.join(save_path, 'val_rs_imgs')):
#         os.mkdir(os.path.join(save_path, 'val_rs_imgs'))
#     if not os.path.exists(os.path.join(save_path, 'val_gray_gt_imgs')):
#         os.mkdir(os.path.join(save_path, 'val_gray_gt_imgs'))
#     if not os.path.exists(os.path.join(save_path, 'val_color_gt_imgs')):
#         os.mkdir(os.path.join(save_path, 'val_color_gt_imgs'))
#
#     print('#################random crop train samples###################')
#     crop_size = 224
#
#     for k in tqdm(list(range(1200)), total=len(list(range(1200))),
#                   leave=False):
#         img_index = np.random.randint(0, len(train_list))
#         rs_img = train_list[img_index][0]
#         gray_gt_img = train_list[img_index][1]
#         color_gt_img = train_list[img_index][2]
#
#         # crop_size_index = np.random.randint(0, 6)
#         # crop_size = crop_sizes[crop_size_index]
#
#         random_h_index = np.random.randint(0, rs_img.shape[0] - crop_size - 1)
#         random_w_index = np.random.randint(0, rs_img.shape[1] - crop_size - 1)
#         crop_rs_img = rs_img[random_h_index:random_h_index + crop_size, random_w_index:random_w_index + crop_size]
#         crop_gray_gt_img = gray_gt_img[random_h_index:random_h_index + crop_size,
#                            random_w_index:random_w_index + crop_size]
#         crop_color_gt_img = color_gt_img[random_h_index:random_h_index + crop_size,
#                             random_w_index:random_w_index + crop_size]
#         imsave(os.path.join(save_path, 'train_rs_imgs', str(k) + '.tif'), crop_rs_img)
#         imsave(os.path.join(save_path, 'train_gray_gt_imgs', str(k) + '.tif'), crop_gray_gt_img)
#         imsave(os.path.join(save_path, 'train_color_gt_imgs', str(k) + '.tif'), crop_color_gt_img)
#
#     print('#################random crop val samples###################')
#
#     crop_size = 224
#     for k in tqdm(list(range(300)), total=len(list(range(300))),
#                   leave=False):
#         img_index = np.random.randint(0, len(val_list))
#         rs_img = val_list[img_index][0]
#         gray_gt_img = val_list[img_index][1]
#         color_gt_img = val_list[img_index][2]
#
#
#
#         random_h_index = np.random.randint(0, rs_img.shape[0] - crop_size - 1)
#         random_w_index = np.random.randint(0, rs_img.shape[1] - crop_size - 1)
#         crop_rs_img = rs_img[random_h_index:random_h_index + crop_size, random_w_index:random_w_index + crop_size]
#         crop_gray_gt_img = gray_gt_img[random_h_index:random_h_index + crop_size,
#                            random_w_index:random_w_index + crop_size]
#         crop_color_gt_img = color_gt_img[random_h_index:random_h_index + crop_size,
#                             random_w_index:random_w_index + crop_size]
#         imsave(os.path.join(save_path, 'val_rs_imgs', 'v' + str(k) + '.tif'), crop_rs_img)
#         imsave(os.path.join(save_path, 'val_gray_gt_imgs', 'v' + str(k) + '.tif'), crop_gray_gt_img)
#         imsave(os.path.join(save_path, 'val_color_gt_imgs', 'v' + str(k) + '.tif'), crop_color_gt_img)
#
#
# def cal_mean_std(datatype,data_path):
#     print("calculate dataset mean and std...")
#     R_means = []
#     G_means = []
#     B_means = []
#     R_stds = []
#     G_stds = []
#     B_stds = []
#     fnames = [os.path.join(data_path, x) for x in os.listdir(data_path)]
#     with tqdm(total=len(fnames)) as pbar:
#         for fname in fnames:
#             im = imread(fname)
#
#             im_R = im[:, :, 0] / 255.0
#             im_G = im[:, :, 1] / 255.0
#             im_B = im[:, :, 2] / 255.0
#             im_R_mean = np.mean(im_R)
#             im_G_mean = np.mean(im_G)
#             im_B_mean = np.mean(im_B)
#             im_R_std = np.std(im_R)
#             im_G_std = np.std(im_G)
#             im_B_std = np.std(im_B)
#             R_means.append(im_R_mean)
#             G_means.append(im_G_mean)
#             B_means.append(im_B_mean)
#             R_stds.append(im_R_std)
#             G_stds.append(im_G_std)
#             B_stds.append(im_B_std)
#
#             pbar.update(1)
#     a = [R_means, G_means, B_means]
#     b = [R_stds, G_stds, B_stds]
#     mean = [0, 0, 0]
#     std = [0, 0, 0]
#     mean[0] = np.mean(a[0])
#     mean[1] = np.mean(a[1])
#     mean[2] = np.mean(a[2])
#     std[0] = np.mean(b[0])
#     std[1] = np.mean(b[1])
#     std[2] = np.mean(b[2])
#     print('数据集{}的RGB平均值为\n[{},{},{}]'.format(datatype,mean[0], mean[1], mean[2]))
#     print('数据集{}的RGB方差为\n[{},{},{}]'.format(datatype,std[0], std[1], std[2]))
#
#
# def calculate_class_pixnum(train_folder):
#     pix_num = dict()
#     imgs = [os.path.join(train_folder, x) for x in os.listdir(train_folder)]
#     for img in tqdm(imgs, total=len(imgs)):
#         im = imread(img)
#         im = np.array(im).astype(np.uint8)
#
#         for row in im:
#             for cell in row:
#                 if int(cell) not in pix_num:
#                     pix_num[int(cell)] = 0
#                 else:
#                     pix_num[int(cell)] = pix_num[int(cell)] + 1
#     print(pix_num)
#
#
# def generate_samples(img_folder, gray_folder, color_folder, sample_num, sample_size, save_path, sample_type='train'):
#     filenames = os.listdir(img_folder)
#     crop_size = sample_size
#     k = 0
#     for filename in filenames:
#         color_file = os.path.join(color_folder, filename)
#         gray_file = os.path.join(gray_folder, filename)
#         img_file = os.path.join(img_folder, filename)
#         color_image = imread(color_file)
#         img_image = imread(img_file)
#         gray_image = imread(gray_file)
#         if sample_type == 'train':
#             if not os.path.exists(os.path.join(save_path, 'train_img_images')):
#                 os.mkdir(os.path.join(save_path, 'train_img_images'))
#             if not os.path.exists(os.path.join(save_path, 'train_gray_images')):
#                 os.mkdir(os.path.join(save_path, 'train_gray_images'))
#             if not os.path.exists(os.path.join(save_path, 'train_color_images')):
#                 os.mkdir(os.path.join(save_path, 'train_color_images'))
#             sample_count = 0
#             while sample_count < sample_num // len(filenames):
#
#                 random_h_index = np.random.randint(0, img_image.shape[0]-crop_size-1)
#                 random_w_index = np.random.randint(0, img_image.shape[1] - crop_size-1)
#                 crop_img_image = img_image[random_h_index:random_h_index+crop_size, random_w_index:random_w_index+crop_size]
#                 crop_gray_image = gray_image[random_h_index:random_h_index + crop_size,
#                                 random_w_index:random_w_index + crop_size]
#                 crop_color_image = color_image[random_h_index:random_h_index + crop_size,
#                                 random_w_index:random_w_index + crop_size]
#                 imsave(os.path.join(save_path, 'train_img_images', 't'+str(k)+'.tif'), crop_img_image)
#                 imsave(os.path.join(save_path, 'train_gray_images', 't'+str(k) + '.tif'), crop_gray_image)
#                 imsave(os.path.join(save_path, 'train_color_images', 't'+str(k) + '.tif'), crop_color_image)
#                 k = k + 1
#                 sample_count = sample_count + 1
#         elif sample_type == 'val':
#             if not os.path.exists(os.path.join(save_path, 'val_img_images')):
#                 os.mkdir(os.path.join(save_path, 'val_img_images'))
#             if not os.path.exists(os.path.join(save_path, 'val_gray_images')):
#                 os.mkdir(os.path.join(save_path, 'val_gray_images'))
#             if not os.path.exists(os.path.join(save_path, 'train_color_images')):
#                 os.mkdir(os.path.join(save_path, 'val_color_images'))
#             for m in  range(img_image.shape[0] // crop_size):
#                 for n in range(img_image.shape[1] // crop_size):
#                     crop_img_image = img_image[m*crop_size:(m+1)*crop_size, n*crop_size:(n+1)*crop_size]
#                     crop_gray_image = gray_image[m*crop_size:(m+1)*crop_size, n*crop_size:(n+1)*crop_size]
#                     crop_color_image = color_image[m*crop_size:(m+1)*crop_size, n*crop_size:(n+1)*crop_size]
#                     imsave(os.path.join(save_path, 'val_img_images', 'v'+str(k)+'.tif'), crop_img_image)
#                     imsave(os.path.join(save_path, 'val_gray_images', 'v'+str(k) + '.tif'), crop_gray_image)
#                     imsave(os.path.join(save_path, 'val_color_images', 'v'+str(k) + '.tif'), crop_color_image)
#                     k = k + 1
#
#
#
#
#
#
#
# if __name__ == "__main__":
#     filenames = os.listdir('/cache/train/color')
#     for filename in filenames:
#         color_to_gray(os.path.join('/cache/train/color', filename),
#                   os.path.join('/cache/train/gray', filename))
#     filenames = os.listdir('/cache/val/color')
#     for filename in filenames:
#         color_to_gray(os.path.join('/cache/val/color', filename),
#                   os.path.join('/cache/val/gray', filename))
#     generate_samples('/cache/train/img', '/cache/train/gray', '/cache/train/color', 3000, 400, '/cache/rs_dataset', 'train')
#     generate_samples('/cache/val/img', '/cache/val/gray', '/cache/val/color', 3000, 400, '/cache/rs_dataset', 'val')
#     cal_mean_std('gg', r'/cache/train/img')
#
#     # calculate_colormap(r'G:\segmentation_data\segmentation_data\guanggu/new_gt_color.tif',
#     #                    r'G:\segmentation_data\segmentation_data\guanggu/new_gt.tif')
#     # split_test_area_new(r'/usr/demo/common_data/seg_data/guanggu/new_img.tif',
#     #                 r'/usr/demo/common_data/seg_data/guanggu/gray_gt.tif',
#     #                 r'/usr/demo/common_data/seg_data/guanggu/new_gt_color.tif',
#     #                 r'/usr/demo/common_data/seg_data/guanggu/dataset/')
#     # calculate_class_pixnum(r'/usr/demo/common_data/seg_data/guanggu/dataset/train_gray_gt_imgs')
#
#
#


from skimage.io import imread, imsave
import os
import random
import shutil
from tqdm import tqdm


def convert_gt(root_path, save_path):
    filenames = os.listdir(root_path)
    for filename in filenames:
        img = imread(os.path.join(root_path, filename))
        # for i in range(img.shape[0]):
        #     for j in range(img.shape[1]):
        #         img[i, j] = img[i, j] // 100 - 1
        img = img // 100 - 1
        imsave(os.path.join(save_path, filename), img)


"""
├── data
│   ├── my_dataset
│   │   ├── img_dir
│   │   │   ├── train
│   │   │   │   ├── xxx{img_suffix}
│   │   │   │   ├── yyy{img_suffix}
│   │   │   │   ├── zzz{img_suffix}
│   │   │   ├── val
│   │   ├── gt_dir
│   │   │   ├── train
│   │   │   │   ├── xxx{seg_map_suffix}
│   │   │   │   ├── yyy{seg_map_suffix}
│   │   │   │   ├── zzz{seg_map_suffix}
│   │   │   ├── val
"""
        
def split_dataset(root_path, dataset_path, img_suffix='.tif', seg_map_suffix='.tif'):
    if not os.path.exists(os.path.join(dataset_path, 'imgs')):
        os.mkdir(os.path.join(dataset_path, 'imgs'))
    if not os.path.exists(os.path.join(dataset_path, 'imgs', 'train')):
        os.mkdir(os.path.join(dataset_path, 'imgs', 'train'))
    if not os.path.exists(os.path.join(dataset_path, 'imgs', 'val')):
        os.mkdir(os.path.join(dataset_path, 'imgs', 'val'))
    if not os.path.exists(os.path.join(dataset_path, 'gts')):
        os.mkdir(os.path.join(dataset_path, 'gts'))
    if not os.path.exists(os.path.join(dataset_path, 'gts', 'train')):
        os.mkdir(os.path.join(dataset_path, 'gts', 'train'))
    if not os.path.exists(os.path.join(dataset_path, 'gts', 'val')):
        os.mkdir(os.path.join(dataset_path, 'gts', 'val'))

    filenames = [x for x in os.listdir(os.path.join(root_path, 'image')) if x.endswith(img_suffix)]
    random.shuffle(filenames)
    
    if os.path.exists(os.path.join(root_path, 'train.txt')):
        train_filenames = []
        with open(os.path.join(root_path, 'train.txt'), 'r') as f:
            train_data = f.readlines()
        for _train_filename in train_data:
            train_filenames.append(_train_filename.strip())
    else:
        train_filenames = filenames[:int(0.8 * len(filenames))]
        with open(os.path.join(root_path, 'train.txt'), 'w') as f:
            for train_filename in train_filenames:
                f.write(train_filename+'\n')
        
    if os.path.exists(os.path.join(root_path, 'val.txt')):
        val_filenames = []
        with open(os.path.join(root_path, 'val.txt'), 'r') as f:
            val_data = f.readlines()
        for _val_filename in val_data:
            val_filenames.append(_val_filename.strip())
    else:
        val_filenames = filenames[int(0.8 * len(filenames)):]
        with open(os.path.join(root_path, 'val.txt'), 'w') as f:
            for val_filename in val_filenames:
                f.write(val_filename+'\n')
    
    
    for filename in tqdm(train_filenames):
        shutil.copy(os.path.join(root_path, 'image', filename), os.path.join(dataset_path, 'imgs', 'train', filename))
        shutil.copy(os.path.join(root_path, 'gt', filename.replace(img_suffix, seg_map_suffix)),
                    os.path.join(dataset_path, 'gts', 'train', filename.replace(img_suffix, seg_map_suffix)))
    
    for filename in tqdm(filenames[int(0.8 * len(filenames)):]):
            
        shutil.copy(os.path.join(root_path, 'image', filename), os.path.join(dataset_path, 'imgs', 'val', filename))
        shutil.copy(os.path.join(root_path, 'gt', filename.replace(img_suffix, seg_map_suffix)),
                    os.path.join(dataset_path, 'gts', 'val', filename.replace(img_suffix, seg_map_suffix)))


if __name__ == "__main__":
    # os.mkdir(r'/cache/train/gt')
    # convert_gt(r'/cache/train/label', r'/cache/train/gt')
    # os.mkdir('/cache/naic_rs')
    split_dataset(r'/storage/dataset/dataset', r'/storage/dataset/wc_dataset')

