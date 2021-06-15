# A Strong Baseline for Image Semantic Segmentation

## Introduction
This project is an open source semantic segmentation toolbox based on PyTorch. It is based on the codes of our Tianchi competition in 2021 (https://tianchi.aliyun.com/competition/entrance/531860/introduction).   
In the competition, our team won the third place (please see [Tianchi_README.md](Tianchi_README.md "Hao Luo")).


### Overview
The master branch works with PyTorch 1.6+.The project now supports popular and contemporary semantic segmentation frameworks, e.g. UNet, DeepLabV3+, HR-Net etc.
### Requirements

## Support
### Backbone
- [x] ResNet (CVPR'2016)
- [x] SeNet (CVPR'2018)
- [x] IBN-Net (CVPR'2018)
- [x] EfficientNet (CVPR'2020)
### Methods
- [x] UNet
- [x] DLink-Net
- [x] Res-UNet
- [x] Efficient-UNet
- [x] Deeplab v3+
- [x] HR-Net
### Tricks
- [x] MixUp /CutMix /CopyPaste
- [x] SWA
- [x] LovaszSoftmax Loss /LargeMarginSoftmax Loss
- [x] FP16
- [x] Multi-scale
### Tools
- [x] large image inference (cut and merge)
- [x] post process (crf/superpixels)
## Quick Start
### Train a model 
```python
python train.py --config_file ${CONFIG_FILE} 
```
- `CONFIG_FILE`: File of training config about model

Examples:   
We trained our model in  Tianchi competition according to the following script:  
Stage 1 (160e)    
```python
python train.py --config_file configs/tc_seg/tc_seg_res_unet_r34_ibn_a_160e.yml
```
Stage 2 (swa 24e)  
```python
python train.py --config_file configs/tc_seg/tc_seg_res_unet_r34_ibn_a_swa.yml
```
### Inference with pretrained models
```python
python inference.py --config_file ${CONFIG_FILE} 
```
- `CONFIG_FILE`: File of inference config about model
### Predict large image with pretrained models
```python
python predict_demo.py --config_file ${CONFIG_FILE} --rs_img_file ${IMAGE_FILE_PATH} --temp_img_save_path ${TEMP_CUT_PATH} -temp_seg_map_save_path ${TEMP_SAVE_PATH} --save_seg_map_file ${SAVE_SEG_FILE} 
```
- `CONFIG_FILE`: File of inference config about model
- `IMAGE_FILE_PATH`: File of large input image to predict
- `TEMP_CUT_PATH`: Temp folder of small cutting samples
- `TEMP_SAVE_PATH`: Temp folder of predict results of cutting samples
- `SAVE_SEG_FILE`: Predict result of the large image


