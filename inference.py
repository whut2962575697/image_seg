from dataset import BaseTestDataImageSet, BaseDataImageSet, get_trm

from config import cfg

from torch.utils.data.dataloader import DataLoader
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import os
from skimage.io import imsave
import numpy as np
import pickle
from utils import AvgerageMeter, make_optimizer, calculate_score, mixup_data, m_evaluate
from prettytable import PrettyTable
import cv2
from skimage import morphology


def get_val_dataloder(cfg, num_gpus):
    val_main_transform = get_trm(cfg, False)
    val_dataset = BaseDataImageSet(cfg, mode='val', img_suffix=cfg.DATASETS.IMG_SUFFIX, seg_map_suffix=cfg.DATASETS.SEG_MAP_SUFFIX,  main_transform=val_main_transform, reduce_zero_label= cfg.DATASETS.REDUCE_ZERO_LABEL
                                   )

    num_workers = cfg.DATALOADER.NUM_WORKERS * num_gpus

    val_loader = DataLoader(
        val_dataset, batch_size=cfg.SOLVER.PER_BATCH, shuffle=False,
        num_workers=num_workers
    )

    return val_loader


def get_test_dataloder(cfg, num_gpus):
    test_main_transform = get_trm(cfg, False)
    test_dataset = BaseTestDataImageSet(cfg, img_suffix=cfg.DATASETS.IMG_SUFFIX, main_transform=test_main_transform)

    num_workers = cfg.DATALOADER.NUM_WORKERS * num_gpus

    test_loader = DataLoader(
        test_dataset, batch_size=cfg.SOLVER.PER_BATCH, shuffle=False,
        num_workers=num_workers
    )

    return test_loader



def calculate_metrics(cfg, data_loder, model, device, save_path, flip_aug=False):
    model.eval()
    conf_mat = np.zeros((cfg.MODEL.N_CLASS, cfg.MODEL.N_CLASS)).astype(np.int64)
    with torch.no_grad():
        for batch in tqdm(data_loder, total=len(data_loder),
                              leave=False):
                # print('aaaaa')
                data, target = batch
                data = data.to(device)
                target = target.to(device)
                outputs = model(data)

                if flip_aug:
                    flip_data = torch.flip(data,[3]) 
                
                    flip_outputs = torch.flip(model(flip_data),[3])
                    
                    outputs = (outputs + flip_outputs) / 2.0
               
                
                target = target.data.cpu()
                outputs = outputs.data.cpu()

                _, preds = torch.max(outputs, 1)
                preds = preds.data.cpu().numpy().squeeze().astype(np.uint8)
                target = target.data.cpu().numpy().squeeze().astype(np.uint8)
                conf_mat += calculate_score(cfg, preds, target, 'val')

              
        val_acc, val_acc_per_class, val_acc_cls, val_IoU, val_mean_IoU, val_kappa = m_evaluate(conf_mat)
        print("val_acc: "+str(val_acc))
        print("val_mean_IoU: "+str(val_mean_IoU))
        print("val_kappa: "+str(val_kappa))
        table = PrettyTable(["序号", "名称", "acc", "IoU"])
        for i in range(cfg.MODEL.N_CLASS):
            table.add_row([i, cfg.DATASETS.CLASS_NAMES[i], val_acc_per_class[i], val_IoU[i]])
        
        print(table)
        # f = open('/cache/score_matrix_train.txt', 'w')
        # f.write(table)
        # f.write("\n")
        # f.write("\n")
        # f.write("\n")
        # f.write("val_acc: "+str(val_acc))
        # f.write("val_mean_IoU: "+str(val_mean_IoU))
        # f.write("val_kappa: "+str(val_kappa))
        # f.close()


        # all_outputs = list()
        # all_targets = list()

        # for batch in tqdm(data_loder, total=len(data_loder),
        #                   leave=False):
        #     data, target = batch
        #     data = data.to(device)
        #     target = target.to(device)
        #     outputs = model(data)


        #     target = target.data.cpu()
        #     outputs = outputs.data.cpu()

        #     all_outputs.append(outputs)
        #     all_targets.append(target)

        # all_outputs = torch.cat(all_outputs, 0)
        # all_targets = torch.cat(all_targets, 0)

        # targets = all_targets.view(-1)
        # outputs = torch.argmax(all_outputs, 1)
        # outputs = outputs.view(-1)

        # acc = accuracy_score(targets, outputs)
        # f1 = f1_score(targets, outputs, average='macro')
        # c_m = confusion_matrix(targets, outputs)

        # f = open(os.path.join(save_path, 'score_matrix_train.txt'), 'w')
        # for i in range(len(c_m)):
        #     for j in range(len(c_m)):
        #         f.write(str(c_m[i][j]) + " ")
        #     f.write("\n")
        # f.write("\n")
        # f.write("\n")
        # f.write("\n")
        # f.write(str(acc)+" ")
        # f.write(str(f1) + " ")
        # f.close()

# def to_categorical(result):
#     return result

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



def inference_samples(data_loder, model, device, save_path, save_logist_path, img_suffix='.tif', seg_map_suffix='.png', flip_aug=False, n_class=8):
   
    model.eval()
    with torch.no_grad():

        for batch in tqdm(data_loder, total=len(data_loder),
                          leave=False):
            data, filenames = batch
            data = data.to(device)


            outputs = model(data)
           
           
            if isinstance(outputs,tuple) or isinstance(outputs,list):
                    outputs = outputs[0]
            

            if flip_aug:
                print("###### flip_aug #######")
                flip_data = torch.flip(data,[3]) 
                flip_outputs = model(flip_data)
                
           
                if isinstance(outputs,tuple) or isinstance(outputs,list):
                    flip_outputs = flip_outputs[0]
                flip_outputs = torch.flip(flip_outputs,[3])
                
                outputs = (outputs + flip_outputs) / 2.0
            
            # for i, filename in enumerate(filenames):
            #     with open(os.path.join(save_logist_path, filename.replace(img_suffix, '.pkl')), 'wb') as f:
            #         pickle.dump(outputs[i].numpy(), f)
            outputs = torch.argmax(outputs, 1)
            post_process = True
            # outputs = outputs + 1
            # outputs = outputs * 100
            # for i, filename in enumerate(filenames):
            #     result = outputs[i].numpy().astype(np.uint8)
            #     imsave(os.path.join(save_path, filename.replace(img_suffix, seg_map_suffix)), result)
            for i, filename in enumerate(filenames):
                # seg_img = np.zeros((256, 256), dtype=np.uint16)
                # pr = outputs[i].numpy()
                # for c in range(n_class):
                #     seg_img[pr[:, :] == c] = int((c + 1) * 100)
                
                if post_process:
                    result = area_connection(outputs[i], 10, 64)
                else:
                    result = outputs[i].data.cpu().numpy().astype(np.uint8)

                seg_img = result + 1
                cv2.imwrite(os.path.join(save_path, filename.replace(img_suffix, seg_map_suffix)), seg_img)



if __name__ == "__main__":
    import argparse
    from common.sync_bn import convert_model
    from model import build_model

    parser = argparse.ArgumentParser(description="Baseline Training")
    parser.add_argument("--config_file", default='', help="path to config file", type=str)
    parser.add_argument("--mode", default="test", help="val/test", type=str)
    parser.add_argument("--save_path", default="test", help="output path", type=str)
    parser.add_argument("--save_logist_path", default="test", help="output path", type=str)
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
        # num_gpus = len(cfg.MODEL.DEVICE_IDS)-1
        num_gpus = torch.cuda.device_count()
        # device_ids = cfg.MODEL.DEVICE_IDS.strip("d")
        # # print(device_ids)
        # device = torch.device("cuda:{0}".format(device_ids))
        # device = torch.device("cuda:0,1")
    else:
        device = 'cpu'
    if args.mode == 'val':
        data_loader = get_val_dataloder(cfg, num_gpus)
    else:
        data_loader = get_test_dataloder(cfg, num_gpus)
    model = build_model(cfg)
    if num_gpus > 1:

        model = torch.nn.DataParallel(model)
        if cfg.SOLVER.SYNCBN:
            model = convert_model(model)


    # param_dict = torch.load(cfg.TEST.WEIGHT)
    param_dict = torch.load(cfg.TEST.WEIGHT, map_location=lambda storage, loc: storage)
    if 'state_dict' in param_dict.keys():
        param_dict = param_dict['state_dict']

            
    start_with_module = False
    for k in param_dict.keys():
        if k.startswith('module.'):
            start_with_module = True
            break
    if start_with_module:
        param_dict = {k[7:] : v for k, v in param_dict.items() }
    print('ignore_param:')
    print([k for k, v in param_dict.items() if k not in model.state_dict() or model.state_dict()[k].size() != v.size()])
    print('unload_param:')
    print([k for k, v in model.state_dict().items() if k not in param_dict.keys() or param_dict[k].size() != v.size()] )

    param_dict = {k: v for k, v in param_dict.items() if k in model.state_dict() and model.state_dict()[k].size() == v.size()}
    for i in param_dict:
        model.state_dict()[i].copy_(param_dict[i])

    # print(param_dict)
    # print(param_dict.keys())
    
    # model.load_state_dict(param_dict)

    model = model.to(device)
    if args.mode == 'val':
        calculate_metrics(cfg, data_loader, model, device, output_dir, flip_aug=False)
        calculate_metrics(cfg, data_loader, model, device, output_dir, flip_aug=True)
    else:
        inference_samples(data_loader, model, device, args.save_path, args.save_logist_path, cfg.DATASETS.IMG_SUFFIX, cfg.DATASETS.SEG_MAP_SUFFIX, flip_aug=cfg.TEST.FLIP_AUG, n_class=cfg.MODEL.N_CLASS)


