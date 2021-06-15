# -*- encoding: utf-8 -*-
'''
@File    :   predict_sh.py    
@Contact :   whut.hexin@foxmail.com
@License :   (C)Copyright 2017-2020, HeXin

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2020/10/16 20:44   xin      1.0         None
'''

import os

# os.system('python predict_demo.py --config_file configs/wc_seg_res_unet_r34_ibn_a_160e.yml '
#           '--rs_img_file /media/dell/E2DE40E3DE40B219/test_samples/largeimg/0.tif '
#           '--temp_img_save_path /media/dell/E2DE40E3DE40B219/test_samples/largeimg/clip_temp0 '
#           '--temp_seg_map_save_path /media/dell/E2DE40E3DE40B219/test_samples/largeimg/seg_temp0  '
#           '--save_seg_map_file /media/dell/E2DE40E3DE40B219/test_samples/largeimg/res_gray1/res0.tif')
#
# os.system('python predict_demo.py --config_file configs/wc_seg_res_unet_r34_ibn_a_160e.yml '
#           '--rs_img_file /media/dell/E2DE40E3DE40B219/test_samples/largeimg/1.tif '
#           '--temp_img_save_path /media/dell/E2DE40E3DE40B219/test_samples/largeimg/clip_temp0 '
#           '--temp_seg_map_save_path /media/dell/E2DE40E3DE40B219/test_samples/largeimg/seg_temp0  '
#           '--save_seg_map_file /media/dell/E2DE40E3DE40B219/test_samples/largeimg/res_gray1/res1.tif')
#
#
# os.system('python predict_demo.py --config_file configs/wc_seg_res_unet_r34_ibn_a_160e.yml '
#           '--rs_img_file /media/dell/E2DE40E3DE40B219/test_samples/largeimg/2.tif '
#           '--temp_img_save_path /media/dell/E2DE40E3DE40B219/test_samples/largeimg/clip_temp0 '
#           '--temp_seg_map_save_path /media/dell/E2DE40E3DE40B219/test_samples/largeimg/seg_temp0  '
#           '--save_seg_map_file /media/dell/E2DE40E3DE40B219/test_samples/largeimg/res_gray1/res2.tif')
#
#
# os.system('python predict_demo.py --config_file configs/wc_seg_res_unet_r34_ibn_a_160e.yml '
#           '--rs_img_file /media/dell/E2DE40E3DE40B219/test_samples/largeimg/3.tif '
#           '--temp_img_save_path /media/dell/E2DE40E3DE40B219/test_samples/largeimg/clip_temp0 '
#           '--temp_seg_map_save_path /media/dell/E2DE40E3DE40B219/test_samples/largeimg/seg_temp0  '
#           '--save_seg_map_file /media/dell/E2DE40E3DE40B219/test_samples/largeimg/res_gray1/res3.tif')




# os.system('python predict_demo.py --config_file configs/wc_hrnet_hr48_160e.yml '
#           '--rs_img_file /media/dell/E2DE40E3DE40B219/test_samples/largeimg/0.tif '
#           '--temp_img_save_path /media/dell/E2DE40E3DE40B219/test_samples/largeimg/clip_temp0 '
#           '--temp_seg_map_save_path /media/dell/E2DE40E3DE40B219/test_samples/largeimg/seg_temp0  '
#           '--save_seg_map_file /media/dell/E2DE40E3DE40B219/test_samples/largeimg/res_gray1/res0.tif')
#
# os.system('python predict_demo.py --config_file configs/wc_hrnet_hr48_160e.yml '
#           '--rs_img_file /media/dell/E2DE40E3DE40B219/test_samples/largeimg/1.tif '
#           '--temp_img_save_path /media/dell/E2DE40E3DE40B219/test_samples/largeimg/clip_temp0 '
#           '--temp_seg_map_save_path /media/dell/E2DE40E3DE40B219/test_samples/largeimg/seg_temp0  '
#           '--save_seg_map_file /media/dell/E2DE40E3DE40B219/test_samples/largeimg/res_gray1/res1.tif')
#
#
# os.system('python predict_demo.py --config_file configs/wc_hrnet_hr48_160e.yml '
#           '--rs_img_file /media/dell/E2DE40E3DE40B219/test_samples/largeimg/2.tif '
#           '--temp_img_save_path /media/dell/E2DE40E3DE40B219/test_samples/largeimg/clip_temp0 '
#           '--temp_seg_map_save_path /media/dell/E2DE40E3DE40B219/test_samples/largeimg/seg_temp0  '
#           '--save_seg_map_file /media/dell/E2DE40E3DE40B219/test_samples/largeimg/res_gray1/res2.tif')
#
#
# os.system('python predict_demo.py --config_file configs/wc_hrnet_hr48_160e.yml '
#           '--rs_img_file /media/dell/E2DE40E3DE40B219/test_samples/largeimg/3.tif '
#           '--temp_img_save_path /media/dell/E2DE40E3DE40B219/test_samples/largeimg/clip_temp0 '
#           '--temp_seg_map_save_path /media/dell/E2DE40E3DE40B219/test_samples/largeimg/seg_temp0  '
#           '--save_seg_map_file /media/dell/E2DE40E3DE40B219/test_samples/largeimg/res_gray1/res3.tif')

os.system('python visualization.py --gray_path /media/dell/E2DE40E3DE40B219/test_samples/largeimg/post_gray --color_path '
          '/media/dell/E2DE40E3DE40B219/test_samples/largeimg/res_color4 ')