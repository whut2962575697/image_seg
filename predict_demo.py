# -*- encoding:utf-8 -*-
# imerage cut
# import tifffile as tif




from skimage.io import imread, imsave
import glob, os


from dataset import BaseTestDataImageSet, BaseDataImageSet, get_trm

from config import cfg

from torch.utils.data.dataloader import DataLoader
import torch
from tqdm import tqdm


from skimage import morphology

import numpy as np
import pickle
import ttach as tta

import cv2


def img_cut(read_path, save_path, imgcHeight, imgcWidth, overlay_padding=56):
    # 获取对需裁剪图像信息
    print('image read...')
    img = imread(read_path)
    img = img.astype(np.uint8)
    imgHeight, imgWidth,imgMode = img.shape
    new_imgHeight,new_imgWidth = imgHeight, imgWidth
    if imgHeight % imgcHeight != 0:
        new_imgHeight = ((imgHeight // imgcHeight)+1)*imgcHeight
    if imgWidth % imgcWidth != 0:
        new_imgWidth = ((imgWidth // imgcWidth) + 1) * imgcWidth
    new_img = np.zeros((new_imgHeight, new_imgWidth, imgMode))
    new_img[:imgHeight, :imgWidth] = img
    img = new_img
    imgHeight, imgWidth = new_imgHeight,new_imgWidth

    print('srcImg shape:', (img.shape))
    print('cutImg shape:',(imgcHeight,imgcWidth,imgMode))
    # 确定裁剪个数
    n_rows = imgHeight // imgcHeight
    n_cols = imgWidth // imgcWidth

    print('cut size:', (n_rows, n_cols))

    # nodata = img[0][0]
    print('load finish')
    for i in range(n_rows):
        for j in range(n_cols):
            # 当前裁剪影像写入
            cur_num = i * n_cols + j
            cur_file = str(cur_num).zfill(5) + '.tif'

            # if imgcHeight+
            # img_cut = np.zeros((1, imgcHeight, imgcWidth, imgMode), np.uint8)

            dy = imgcHeight * i
            dx = imgcWidth * j

            if dy-overlay_padding>=0 and dy+imgcHeight+overlay_padding<=imgHeight and dx-overlay_padding>=0 \
                    and dx+imgcWidth+overlay_padding<=imgWidth:
                img_cut = img[dy-overlay_padding:dy+imgcHeight+overlay_padding,
                          dx-overlay_padding:dx+imgcWidth+overlay_padding]
            else:
                if dy-overlay_padding<0:
                    sy = dy
                    ey = dy+imgcHeight+2*overlay_padding
                else:
                    sy = dy-overlay_padding
                    if dy+imgcHeight+overlay_padding>imgHeight:
                        ey = imgHeight
                        sy = imgHeight-imgcHeight-2*overlay_padding
                    else:
                        ey = dy+imgcHeight+overlay_padding


                if dx-overlay_padding<0:
                    sx = dx
                    ex = dx+imgcWidth+2*overlay_padding

                else:
                    sx = dx-overlay_padding
                    if dx+imgcWidth+overlay_padding>imgWidth:
                        ex = imgWidth
                        sx = imgWidth-imgcWidth-2*overlay_padding
                    else:
                        ex = dx+imgcWidth+overlay_padding

                # print(sy, ey, sx, ex)
                img_cut = img[sy:ey, sx:ex]
                # img_cut = np.zeros((imgcHeight+2*overlay_padding, imgcWidth+2*overlay_padding, imgMode), np.uint8)
                # img_cut[overlay_padding:imgcHeight+overlay_padding, overlay_padding:imgcWidth+overlay_padding] = img[dy:dy + imgcHeight, dx:dx + imgcWidth]

            # for y in range(imgcHeight):
            #     for x in range(imgcWidth):
            #
            #         if y + j * imgcWidth < imgWidth and x + i * imgcHeight < imgHeight:
            #             img_cut[0, y, x] = img[y + dy, x + dx]
            #         else:
            #             img_cut[0, y, x] = nodata
            #             print("aaaaaa")
            # print(img_cut.shape)
            imsave(os.path.join(save_path, cur_file), img_cut)

        print('cut rows:' + str(i + 1) + '/' + str(n_rows))
    print('cut finish')


def image_m(o_img, read_path, save_path, imgcHeight, imgcWidth, overlay_padding=56):
    # ！！！此处为需要修改参数！！！#
    # 设置拼接图像的尺寸,及读写路径
    # read_path = '15_clip/result/'
    # save_path = '15_clip/r/img_merage.tif'


    # # 预先读取基本图像尺寸
    img = imread(o_img)
    img = img.astype(np.uint8)
    [img_height,img_width, imgMode] = img.shape
    print('o_img_shape:',img.shape)
    o_img_shape  =img.shape
    new_imgHeight, new_imgWidth = img_height, img_width
    if img_height % imgcHeight != 0:
        new_imgHeight = ((img_height // imgcHeight) + 1) * imgcHeight
    if img_width % imgcWidth != 0:
        new_imgWidth = ((img_width // imgcWidth) + 1) * imgcWidth
    # new_img = np.zeros((new_imgHeight, new_imgWidth, imgMode))
    # new_img[:img_height, :img_width] = img
    # img = new_img
    img_height, img_width = new_imgHeight, new_imgWidth



    n_rows = img_height//imgcHeight
    n_cols = img_width//imgcWidth


    print('merage size:', (n_rows, n_cols))

    # 生成拼接图像数组
    dstImg = np.zeros((imgcHeight*n_rows, imgcWidth*n_cols), np.uint8)
    print('dstImg shape:',dstImg.shape)
    print('load start')

    #逐个读取图像,先确定小图像位置,在确定小图像像素点与大图像像素点坐标关系
    # 左上原点横向读取
    for i in range(n_rows):
        for j in range(n_cols):
            cur = i*n_cols + j
            cur_file = str(cur).zfill(5) + '.tif'
            img = imread(os.path.join(read_path, cur_file))

            dy = i * imgcHeight
            dx = j * imgcWidth
            # dstImg[dy:dy + imgcHeight, dx:dx + imgcWidth] = img[overlay_padding:overlay_padding + imgcHeight,
            #                                                 overlay_padding:overlay_padding + imgcWidth]
            if dy - overlay_padding >= 0 and dy + imgcHeight + overlay_padding <= img_height and dx - overlay_padding >= 0 \
                    and dx + imgcWidth + overlay_padding <= img_width:
                dstImg[dy:dy+imgcHeight, dx:dx+imgcWidth] = img[overlay_padding:overlay_padding+imgcHeight,
                                                            overlay_padding:overlay_padding+imgcWidth]
            else:

                if dy - overlay_padding < 0:
                    sy = 0
                    ey = imgcHeight
                else:
                    sy = overlay_padding
                    if dy + imgcHeight + overlay_padding > img_height:
                        ey = imgcHeight + overlay_padding
                        sy = overlay_padding
                    else:
                        ey = imgcHeight + overlay_padding

                if dx - overlay_padding < 0:
                    sx = 0
                    ex = imgcWidth
                else:
                    sx = overlay_padding
                    if dx + imgcWidth + overlay_padding > img_width:
                        ex = imgcWidth + overlay_padding
                        sx = overlay_padding
                    else:
                        ex = imgcWidth + overlay_padding
                # print(sy, ey, sx, ex)

                dstImg[dy:dy + imgcHeight, dx:dx + imgcWidth] = img[sy:ey, sx:ex]


        print('write:'+str(i+1)+'/'+str(n_rows))
    dstImg = dstImg[:o_img_shape[0],:o_img_shape[1]]
    imsave(save_path,dstImg)
    print('merage finish')


def get_test_dataloder(cfg, num_gpus):
    test_main_transform = get_trm(cfg, False)
    test_dataset = BaseTestDataImageSet(cfg, img_suffix=cfg.DATASETS.IMG_SUFFIX, main_transform=test_main_transform)

    num_workers = cfg.DATALOADER.NUM_WORKERS * num_gpus

    test_loader = DataLoader(
        test_dataset, batch_size=cfg.SOLVER.PER_BATCH, shuffle=False,
        num_workers=num_workers
    )

    return test_loader


def to_categorical(label, N):
    size = list(label.size())
    label = label.view(-1)   # reshape 为向量
    ones = torch.eye(N).cuda()
    ones = ones.index_select(0, label)   # 用上面的办法转为换one hot
    size.append(N)  # 把类别输目添到size的尾后，准备reshape回原来的尺寸
    return ones.view(*size).cpu().numpy().astype(np.uint8)

def area_connection(result, n_class, area_threshold):
    result = to_categorical(result, n_class)
    for i in tqdm(range(n_class)):
        result[:, :, i] = morphology.remove_small_objects(result[:, :, i]==1, min_size=area_threshold,connectivity=1, in_place=True)
        result[:, :, i] = morphology.remove_small_holes(result[:, :, i]==1, min_size=area_threshold,connectivity=1,in_place=True)
    result = np.argmax(result,axis=2).astype(np.uint8)
    return result




def inference_samples(data_loder, model, device, save_path, img_suffix='.tif', seg_map_suffix='.png', flip_aug=False, n_class=8, save_logist_path=None):
    transforms = tta.Compose([
        tta.HorizontalFlip(),
        tta.Rotate90(angles=[0,90,180,270]),
    ])
    use_tta = True
    model.eval()
    with torch.no_grad():

        for batch in tqdm(data_loder, total=len(data_loder),
                          leave=False):
            data, filenames = batch
            data = data.to(device)

            if use_tta:
                outputs = None
                for transformer in transforms:
                    aug_img = transformer.augment_image(data)
                    model_output = model(aug_img)
                    deaug_mask = transformer.deaugment_mask(model_output)
                    if outputs is not None:
                        outputs = outputs + deaug_mask
                    else:
                        outputs = deaug_mask
                outputs = outputs/ (1.0*len(transforms))
            else:
                outputs = model(data)


            # if flip_aug:
            #     flip_data = torch.flip(data,[3])
            #
            #     flip_outputs = torch.flip(model(flip_data),[3])
            #
            #     outputs = (outputs + flip_outputs) / 2.0
            # outputs = outputs.data.cpu()
            if save_logist_path:
                for i, filename in enumerate(filenames):
                    with open(os.path.join(save_logist_path, filename.replace(img_suffix, '.pkl')), 'wb') as f:
                        pickle.dump(outputs[i].cpu().numpy(), f)

            outputs = torch.argmax(outputs, 1)

            post_process = True
            for i, filename in enumerate(filenames):

                if post_process:
                    result = area_connection(outputs[i], n_class, 64)
                else:
                    result = outputs[i].data.cpu().numpy().astype(np.uint8)

                # result = outputs[i].numpy().astype(np.uint8)
                imsave(os.path.join(save_path, filename.replace(img_suffix, seg_map_suffix)), result)
    

if __name__ == "__main__":
    import argparse
    from common.sync_bn import convert_model
    from model import build_model

    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    parser = argparse.ArgumentParser(description="Baseline Training")
    parser.add_argument("--config_file", default='', help="path to config file", type=str)
    # parser.add_argument("--mode", default="test", help="val/test", type=str)
    parser.add_argument("--rs_img_file", default="img_file", help="rs file path", type=str)
    parser.add_argument("--temp_img_save_path", default="img_file", help="temp img save path", type=str)
    parser.add_argument("--temp_seg_map_save_path", default="seg_map_file", help="seg map save path", type=str)
    parser.add_argument("--save_seg_map_file", default="test", help="seg map file path", type=str)
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    num_gpus = 0
    device = cfg.MODEL.DEVICE
    if cfg.MODEL.DEVICE == 'cuda' and torch.cuda.is_available():
        num_gpus = 1
        # num_gpus = len(cfg.MODEL.DEVICE_IDS.split(','))
        # device_ids = cfg.MODEL.DEVICE_IDS.strip("d")
        # # print(device_ids)
        # device = torch.device("cuda:{0}".format(device_ids))
        # device = torch.device("cuda:0,1")
    else:
        device = 'cpu'
    # cut image
    img_cut(args.rs_img_file, args.temp_img_save_path, cfg.INPUT.SIZE_TEST[0], cfg.INPUT.SIZE_TEST[1])
    
    data_loader = get_test_dataloder(cfg, num_gpus)
    model = build_model(cfg)
    if num_gpus > 1:

        model = torch.nn.DataParallel(model)
        if cfg.SOLVER.SYNCBN:
            model = convert_model(model)


    param_dict = torch.load(cfg.TEST.WEIGHT, map_location=lambda storage, loc: storage)

    if 'state_dict' in param_dict.keys():
        param_dict = param_dict['state_dict']

    start_with_module = False
    for k in param_dict.keys():
        if k.startswith('module.'):
            start_with_module = True
            break

    if start_with_module:
        param_dict = {k[7:]: v for k, v in param_dict.items() if k.startswith('module.')}
        # param_dict = {k[7:]: v for k, v in param_dict.items() if k.startswith('module.')}
        # print(param_dict.keys())
    print('ignore_param:')
    print([k for k, v in param_dict.items() if k not in model.state_dict() or
           model.state_dict()[k].size() != v.size()])
    print('unload_param:')
    print([k for k, v in model.state_dict().items() if k not in param_dict.keys() or
           param_dict[k].size() != v.size()])

    param_dict = {k: v for k, v in param_dict.items() if k in model.state_dict() and
                  model.state_dict()[k].size() == v.size()}
    for i in param_dict:
        model.state_dict()[i].copy_(param_dict[i])



    # model.load_state_dict(param_dict)

    model = model.to(device)
    inference_samples(data_loader, model, device, args.temp_seg_map_save_path, cfg.DATASETS.IMG_SUFFIX, cfg.DATASETS.SEG_MAP_SUFFIX,  cfg.TEST.FLIP_AUG, cfg.MODEL.N_CLASS)
    image_m(args.rs_img_file, args.temp_seg_map_save_path, args.save_seg_map_file, cfg.INPUT.SIZE_TEST[0], cfg.INPUT.SIZE_TEST[1])