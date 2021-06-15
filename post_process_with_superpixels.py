from skimage.io import imread, imsave
import numpy as np

def post_process(predict_img_file, roi_img_file, save_file, max_superpix_num=20000, cls_num=13):
    roi_img = imread(roi_img_file)
    predict_img = imread(predict_img_file)
    new_img = np.zeros(predict_img.shape).astype(np.int8)
    for i in range(max_superpix_num):
        pix_nums = [0]*cls_num
        for j in range(cls_num):
           pix_num = np.sum(predict_img[roi_img == i] == j)
           # print(pix_num)
           pix_nums[j] = pix_num
        # print(np.argmax(pix_nums))
        if pix_nums[5]/(1.0*np.sum(pix_nums))>0.3:
            new_img[roi_img == i] = 5
        else:
            new_img[roi_img == i] = np.argmax(pix_nums)
        # print(new_img)
    imsave(save_file, new_img)


if __name__ == "__main__":
    # post_process(r'/media/dell/E2DE40E3DE40B219/test_samples/largeimg/res_gray1/res0.tif',
    #              r'/media/dell/E2DE40E3DE40B219/superpixel_seg/40/0.tif',
    #              r'/media/dell/E2DE40E3DE40B219/test_samples/largeimg/post_gray/res0.tif')
    post_process(r'/media/dell/E2DE40E3DE40B219/test_samples/largeimg/res_gray1/res1.tif',
                 r'/media/dell/E2DE40E3DE40B219/superpixel_seg/40/1.tif',
                 r'/media/dell/E2DE40E3DE40B219/test_samples/largeimg/post_gray/res1.tif')
    post_process(r'/media/dell/E2DE40E3DE40B219/test_samples/largeimg/res_gray1/res2.tif',
                 r'/media/dell/E2DE40E3DE40B219/superpixel_seg/40/2.tif',
                 r'/media/dell/E2DE40E3DE40B219/test_samples/largeimg/post_gray/res2.tif')
    post_process(r'/media/dell/E2DE40E3DE40B219/test_samples/largeimg/res_gray1/res3.tif',
                 r'/media/dell/E2DE40E3DE40B219/superpixel_seg/40/3.tif',
                 r'/media/dell/E2DE40E3DE40B219/test_samples/largeimg/post_gray/res3.tif')
    #
    #
