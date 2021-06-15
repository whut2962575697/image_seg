# -*- encoding: utf-8 -*-
'''
@File    :   train_sh.py    
@Contact :   whut.hexin@foxmail.com
@License :   (C)Copyright 2017-2020, HeXin

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2020/9/9 15:38   xin      1.0         None
'''

import os

os.system('python train.py --config_file configs/wc_hrnet_hr48_160e.yml')
# os.system('python train.py --config_file configs/wc_seg_res_unet_r34_ibn_a_160e.yml')
# os.system('python train.py --config_file configs/wc_seg_res_unet_r34_ibn_a_swa.yml')