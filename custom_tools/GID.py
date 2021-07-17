from skimage.io import imread, imsave
# import os
#
# color_key_list = [
#     {'color':(0, 200, 0), 'label': '林地', 'cls_num': 1, 'cls_name': 'forest'},
#     {'color':(0, 255, 255), 'label': '草地', 'cls_num': 2, 'cls_name': 'grass'},
#     {'color':(150, 200, 150), 'label': '耕地', 'cls_num': 3, 'cls_name': 'farmland'},
#     {'color':(200, 0, 200), 'label': '水域', 'cls_num': 4, 'cls_name': 'water'},
#     {'color':(150, 0, 250), 'label': '道路', 'cls_num': 5, 'cls_name': 'road'},
#     {'color':(150, 150, 250), 'label': '城镇建设用地', 'cls_num': 6, 'cls_name': 'urban_area'},
#     {'color':(250, 200, 0), 'label': '农村建设用地', 'cls_num': 7, 'cls_name': 'countryside'},
#     {'color':(25, 25, 112), 'label': '工业用地', 'cls_num': 8, 'cls_name': 'industrial_land'},
#     {'color':(255, 0, 0), 'label': '构筑物', 'cls_num': 9, 'cls_name': 'construction'},
#     {'color':(255, 20, 147), 'label': '裸地', 'cls_num': 10, 'cls_name': 'bareland'},
#     {'color':(255, 228, 181), 'label': '密林', 'cls_num': 11, 'cls_name': 'denseForest'},
#     {'color':(0, 134, 139), 'label': '疏林', 'cls_num': 12, 'cls_name': 'openForest'},
#     {'color':(150, 250, 0), 'label': 'others', 'cls_num': 13, 'cls_name': 'others'},
#                  ]


import json, os
import numpy as np
import tifffile as tiff
from PIL import Image
from tqdm import tqdm


def get_label_from_palette(label_img, palette_file='Palette.json'):

    text = { "0": [0,200,0],
  "1": [150,250,0],
  "2": [150,200,150],
  "3": [200,0,200],
  "4": [150,0,250],
  "5": [150,150,250],
  "6": [250,200,0],
  "7": [200,200,0],
  "8": [200,0,0],
  "9": [250,0,150],
  "10": [200,150,150],
  "11": [250,150,150],
  "12": [0,0,200],
  "13": [0,150,200],
  "14": [0,200,250],
  "15": [0,0,0]
}
    label = np.zeros((label_img.shape[0], label_img.shape[1]), dtype=np.uint8)
    for i in range(label_img.shape[0]):
        print(i)
        for j in range(label_img.shape[1]):
            assert list(label_img[i, j, :]) in list(text.values())

            label[i, j] = int(list(text.keys())[list(text.values()).index(list(label_img[i, j, :]))])

        return label






def main(path):
    for pic in os.listdir(path):
        if 'label' in pic:
            print(pic)

            # ---- read RGB label
            label = Image.open(path + '/' + pic)
            label = np.asarray(label)
            # ----- another way
            # label = tiff.imread(path + '/' +pic)  # the code of this line can be run in windows system, which in ubuntu will get a error !!

            label = get_label_from_palette(label)
            tiff.imsave(path + '/' + pic[:-9] + 'new-L.tif', label)


def cut_samples(img_path, gt_path, img_save, gt_save, image_size,mode='train', overlay_padding=56):
    count = 0
    for filename in tqdm([x for x in os.listdir(img_path) if x.endswith('tif')]):
        img = imread(os.path.join(img_path, filename))
        gt = imread(os.path.join(gt_path, filename.replace('.tif', '_label.tif')))
        imgHeight, imgWidth, imgMode = img.shape
        new_imgHeight, new_imgWidth = imgHeight, imgWidth
        if img.shape[0] % image_size != 0:
            new_imgHeight = ((img.shape[0] // image_size) + 1) * image_size
        if img.shape[1] % image_size != 0:
            new_imgWidth = ((img.shape[1] // image_size) + 1) * image_size
        new_img = np.zeros((new_imgHeight, new_imgWidth, imgMode)).astype(np.uint8)
        new_img[:imgHeight, :imgWidth] = img
        img = new_img

        new_gt = np.ones((new_imgHeight, new_imgWidth)).astype(np.uint8)*255
        new_gt[:imgHeight, :imgWidth] = gt
        gt = new_gt





        if mode != 'train':
            # 切割方式 顺序切割
            w_c = img.shape[0] // image_size
            h_c = img.shape[1] // image_size
            for i in range(w_c):
                for j in range(h_c):
                    sample_img = img[i*image_size:(i+1)*image_size, j*image_size:(j+1)*image_size]
                    sample_gt = gt[i*image_size:(i+1)*image_size, j*image_size:(j+1)*image_size]
                    imsave(os.path.join(img_save, str(count)+'.tif'), sample_img)
                    imsave(os.path.join(gt_save, str(count) + '.tif'), sample_gt)
                    count = count + 1
        else:
            # 重叠切割
            w_c = img.shape[0] // image_size
            h_c = img.shape[1] // image_size
            for i in range(w_c):
                for j in range(h_c):

                    dy = image_size * i
                    dx = image_size * j

                    if dy - overlay_padding >= 0 and dy + image_size + overlay_padding <= img.shape[0] and dx - overlay_padding >= 0 \
                            and dx + image_size + overlay_padding <= img.shape[1]:
                        sample_img = img[dy - overlay_padding:dy + image_size + overlay_padding,
                                  dx - overlay_padding:dx + image_size + overlay_padding]
                        sample_gt = gt[dy - overlay_padding:dy + image_size + overlay_padding,
                                  dx - overlay_padding:dx + image_size + overlay_padding]
                    else:
                        if dy - overlay_padding < 0:
                            sy = dy
                            ey = dy + image_size + 2 * overlay_padding
                        else:
                            sy = dy - overlay_padding
                            if dy + image_size + overlay_padding > img.shape[0]:
                                ey = img.shape[0]
                                sy = img.shape[0] - image_size - 2 * overlay_padding
                            else:
                                ey = dy + image_size + overlay_padding

                        if dx - overlay_padding < 0:
                            sx = dx
                            ex = dx + image_size + 2 * overlay_padding

                        else:
                            sx = dx - overlay_padding
                            if dx + image_size + overlay_padding > img.shape[1]:
                                ex = img.shape[1]
                                sx = img.shape[1] - image_size - 2 * overlay_padding
                            else:
                                ex = dx + image_size + overlay_padding

                        sample_img = img[sy:ey, sx:ex]
                        sample_gt = gt[sy:ey, sx:ex]
                    # sample_img = img[i * image_size:(i + 1) * image_size, j * image_size:(j + 1) * image_size]
                    # sample_gt = gt[i * image_size:(i + 1) * image_size, j * image_size:(j + 1) * image_size]
                    imsave(os.path.join(img_save, str(count) + '.tif'), sample_img)
                    imsave(os.path.join(gt_save, str(count) + '.tif'), sample_gt)
                    count = count + 1


if __name__ == '__main__':

    from skimage.io import imsave, imread
    # from tifffile import imread, im
    import cv2
    import os
    import numpy as np

    # color_dict = [
    #     (0, 200, 0),
    #     (150, 250, 0),
    #     (150, 200, 150),
    #     (200, 0, 200),
    #     (150, 0, 250),
    #     (150, 150, 250),
    #     (250, 200, 0),
    #     (200, 200, 0),
    #     (200, 0, 0),
    #     (250, 0, 150),
    #     (200, 150, 150),
    #     (250, 150, 150),
    #     (0, 0, 200),
    #     (0, 150, 200),
    #     (0, 200, 250),
    #     (0, 0, 0)
    # ]
    #
    # gt_path = r'E:\rssrai2019_semantic_segmentation\train\train'
    # save_path = r'E:\rssrai2019_semantic_segmentation\train\gray_path'
    #
    # filenames = [_ for _ in os.listdir(gt_path) if _.endswith('_label.tif')]
    # for filename in filenames:
    #     print(filename)
    #     color_gt = imread(os.path.join(gt_path, filename))
    #     gray_gt = np.ones(color_gt.shape[:2]).astype(np.uint8) * 255
    #     for i in range(len(color_dict)):
    #         gray_gt[(color_gt[:, :, 0] == color_dict[i][0]) * (color_gt[:, :, 1] == color_dict[i][1]) * (
    #                     color_gt[:, :, 2] == color_dict[i][2])] = i
    #     # gray_gt[gray_gt == 15] = 255
    #     imsave(os.path.join(save_path, filename), gray_gt)
    # train_path = r'E:\rssrai2019_semantic_segmentation\train\train'
    # val_path = r'E:\rssrai2019_semantic_segmentation\val\val'
    # main(train_path)
    # main(val_path)

    os.mkdir('/cache/dataset')
    os.mkdir('/cache/dataset/train')
    os.mkdir('/cache/dataset/train/imgs')
    os.mkdir('/cache/dataset/train/gts')
    os.mkdir('/cache/dataset/val')
    os.mkdir('/cache/dataset/val/imgs')
    os.mkdir('/cache/dataset/val/gts')

    cut_samples(r'/cache/GID/train/img_path', r'/cache/GID/train/gray_path',
                r'/cache/dataset/train/imgs', r'/cache/dataset/train/gts', 128, mode='train', overlay_padding=64)

    cut_samples(r'/cache/GID/val/img_path',
                r'/cache/GID/val/gray_path',
                r'/cache/dataset/val/imgs',
                r'/cache/dataset/val/gts', 256, mode='val', overlay_padding=0)

    os.mkdir('/home/ma-user/work/dataset/GID_DATASET')
    os.mkdir('/home/ma-user/work/dataset/GID_DATASET/train')
    os.mkdir('/home/ma-user/work/dataset/GID_DATASET/train/imgs')
    os.mkdir('/home/ma-user/work/dataset/GID_DATASET/train/gts')
    os.mkdir('/home/ma-user/work/dataset/GID_DATASET/val')
    os.mkdir('/home/ma-user/work/dataset/GID_DATASET/val/imgs')
    os.mkdir('/home/ma-user/work/dataset/GID_DATASET/val/gts')

    cut_samples(r'/home/ma-user/work/dataset/GID/train/img_path', r'/home/ma-user/work/dataset/GID/train/gray_path',
                r'/home/ma-user/work/dataset/GID_DATASET/train/imgs', r'/home/ma-user/work/dataset/GID_DATASET/train/gts', 128, mode='train', overlay_padding=64)

    cut_samples(r'/home/ma-user/work/dataset/GID/val/img_path',
                r'/home/ma-user/work/dataset/GID/val/gray_path',
                r'/home/ma-user/work/dataset/GID_DATASET/val/imgs',
                r'/home/ma-user/work/dataset/GID_DATASET/val/gts', 256, mode='val', overlay_padding=0)