import shutil
import os
from tqdm import tqdm
import random

# make dataset
root_path = r'../../tcdata/suichang_round1_train_210120'
rs_save_path = r'../../user_data/temp_data/tc_data/imgs'
seg_map_save_path = r'../../user_data/temp_data/tc_data/gts'

img_files = [x for x in os.listdir(root_path) if x.endswith('tif')]

for filename in tqdm(img_files):
    rs_img_file = os.path.join(root_path, filename)
    mask_img_file = os.path.join(root_path, filename.replace('tif', 'png'))

    shutil.copy(rs_img_file, os.path.join(rs_save_path, filename))
    shutil.copy(mask_img_file, os.path.join(seg_map_save_path, filename.replace('tif', 'png')))


# make dataset
root_path = r'../../tcdata/suichang_round2_train_210316'
rs_save_path = r'../../user_data/temp_data/tc_data/imgs'
seg_map_save_path = r'../../user_data/temp_data/tc_data/gts'

img_files = [x for x in os.listdir(root_path) if x.endswith('tif')]

for filename in tqdm(img_files):
    rs_img_file = os.path.join(root_path, filename)
    mask_img_file = os.path.join(root_path, filename.replace('tif', 'png'))

    shutil.copy(rs_img_file, os.path.join(rs_save_path, filename.replace('.tif', '_new.tif')))
    shutil.copy(mask_img_file, os.path.join(seg_map_save_path, filename.replace('.tif', '_new.png')))

# split trainval
rs_imgs = os.listdir(rs_save_path)

random.shuffle(rs_imgs)

train_rs_path = r'../../user_data/temp_data//split_dataset/train/imgs'
train_gt_path = r'../../user_data/temp_data/split_dataset/train/gts'
for filename in rs_imgs[:int(1*len(rs_imgs))]:
    shutil.copy(os.path.join(rs_save_path, filename), os.path.join(train_rs_path, filename))
    shutil.copy(os.path.join(seg_map_save_path, filename.replace('tif', 'png')), os.path.join(train_gt_path, filename.replace('tif', 'png')))

val_rs_path = r'../../user_data/temp_data//split_dataset/val/imgs'
val_gt_path = r'../../user_data/temp_data//split_dataset/val/gts'
for filename in rs_imgs[int(0.8*len(rs_imgs)):]:
    shutil.copy(os.path.join(rs_save_path, filename), os.path.join(val_rs_path, filename))
    shutil.copy(os.path.join(seg_map_save_path, filename.replace('tif', 'png')), os.path.join(val_gt_path, filename.replace('tif', 'png')))