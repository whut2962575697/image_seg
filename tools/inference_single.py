'''
server中使用GPU做infer
面向工业瓷砖大赛demo
'''
from ai_hub import inferServer
import json
import torch
import torch.nn as nn
import cv2
import numpy as np


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

import json
import base64
import cv2
from PIL import Image
from io import BytesIO
import numpy as np
import torch
from torch.autograd import Variable as V
import base64


class myserver(inferServer):
    def __init__(self, model):
        super().__init__(model)
        print("init_myserver")
        self.main_transform = get_trm(cfg, False)
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.device = device
        # self.model = model.to(device)
        self.model = model
        self.model.eval()

    def pre_process(self, data):
        print("my_pre_process.")
        # json process
        # file example
        json_data = json.loads(data.get_data().decode('utf-8'))
        img = json_data.get("img")
        bast64_data = img.encode(encoding='utf-8')
        img = base64.b64decode(bast64_data)
        bytesIO = BytesIO()
        img = Image.open(BytesIO(bytearray(img)))
        img = np.array(img)
        # img = img.astype(np.float32)
        data = {'image': img}
        aug = self.main_transform(**data)
        img = aug['image']

        img = V(img.cuda())
        return img


    #pridict default run as follow：
    def pridect(self, data):
        # ret = self.model(data)
        with torch.no_grad():
            data = data.unsqueeze(0)
            flip_data = torch.flip(data,[3])
            data = torch.cat([data, flip_data], 0)
            ret = self.model(data)
            ret = torch.mean(ret, 0)
            _, ret = torch.max(ret, 1)

        return ret

    def post_process(self, processed_data):
        processed_data = processed_data.cpu().data.numpy()
        img_encode = np.array(cv2.imencode('.png', processed_data)[1]).tobytes()
        bast64_data = base64.b64encode(img_encode)
        bast64_str = str(bast64_data, 'utf-8')
        return bast64_str


# 赛题方数据同学的output2result函数供大家参考，infer_score_thre为自定义阈值
# def output2result(result, name, infer_score_thre):
#     image_name = name
#     predict_rslt = []
#     for i, res_perclass in enumerate(result):
#         class_id = i + 1
#         for per_class_results in res_perclass:
#             xmin, ymin, xmax, ymax, score = per_class_results
#             if score < infer_score_thre:
#                 continue
#             xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
#             dict_instance = dict()
#             dict_instance['name'] = image_name
#             dict_instance['category'] = class_id
#             dict_instance["score"] = round(float(score), 6)
#             dict_instance["bbox"] = [xmin, ymin, xmax, ymax]
#             predict_rslt.append(dict_instance)
#
#     return predict_rslt


if __name__ == '__main__':
    # mymodel = mymodel()
    from common.sync_bn import convert_model
    from model import build_model



    config_file = r''
    checkpoint_file = r''

    cfg.merge_from_list(config_file)
    cfg.freeze()

    num_gpus = 0
    device = 'cuda'
    if torch.cuda.is_available():
        # num_gpus = len(cfg.MODEL.DEVICE_IDS)-1
        num_gpus = torch.cuda.device_count()
        # device_ids = cfg.MODEL.DEVICE_IDS.strip("d")
        # # print(device_ids)
        # device = torch.device("cuda:{0}".format(device_ids))
        # device = torch.device("cuda:0,1")
    else:
        device = 'cpu'

    model = build_model(cfg)
    if num_gpus > 1:

        model = torch.nn.DataParallel(model)
        if cfg.SOLVER.SYNCBN:
            model = convert_model(model)

    # param_dict = torch.load(cfg.TEST.WEIGHT)
    param_dict = torch.load(checkpoint_file, map_location=lambda storage, loc: storage)
    if 'state_dict' in param_dict.keys():
        param_dict = param_dict['state_dict']

    # start_with_module = False
    # for k in param_dict.keys():
    #     if k.startswith('module.'):
    #         start_with_module = True
    #         break
    # if start_with_module:
    #     param_dict = {k[7:]: v for k, v in param_dict.items()}
    print('ignore_param:')
    print([k for k, v in param_dict.items() if
           k not in model.state_dict() or model.state_dict()[k].size() != v.size()])
    print('unload_param:')
    print([k for k, v in model.state_dict().items() if
           k not in param_dict.keys() or param_dict[k].size() != v.size()])

    param_dict = {k: v for k, v in param_dict.items() if
                  k in model.state_dict() and model.state_dict()[k].size() == v.size()}
    for i in param_dict:
        model.state_dict()[i].copy_(param_dict[i])




    model = model.to(device)
    myserver = myserver(model)
    # run your server, defult ip=localhost port=8080 debuge=false
    myserver.run(debuge=True)  # myserver.run("127.0.0.1", 1234)