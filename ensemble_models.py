import pickle
import os
import numpy as np
from skimage.io import imsave, imread
import cv2
from prettytable import PrettyTable

from utils.metrics import confusion_matrix
from utils.metrics import evaluate as m_evaluate


def ensemble(logist_paths, save_path, seg_map_suffix, gt_path=None, n_class=8, class_names = ['st', 'jyts', 'jz', 'gd', 'cd', 'ld', 'lt', 'qt']):
    conf_mat = np.zeros((n_class, n_class)).astype(np.int64)
    filenames = os.listdir(logist_paths[0])
    for filename in filenames:
        with open(os.path.join(logist_paths[0], filename), 'rb') as f:
                logist_output = pickle.load(f)
        for logist_path in logist_paths[1:]:
            with open(os.path.join(logist_path, filename), 'rb') as f:
                output = pickle.load(f)
            logist_output = logist_output + output
        logist_output = logist_output / (1.0*len(logist_paths))
        pr = np.argmax(logist_output, 0)

        if gt_path:
            target = imread(os.path.join(gt_path, filename.replace('pkl', 'png')))

            preds = pr.astype(np.uint8)
            target = target.astype(np.uint8)
            conf_mat += confusion_matrix(pred=preds.flatten(),
                                                label=target.flatten(),
                                                num_classes=n_class)

        seg_img = np.zeros((256, 256), dtype=np.uint16)

        for c in range(n_class):
            seg_img[pr[:, :] == c] = int((c + 1) * 100)
        cv2.imwrite(os.path.join(save_path, filename.replace('pkl', 'png')), seg_img)
    if gt_path:
        val_acc, val_acc_per_class, val_acc_cls, val_IoU, val_mean_IoU, val_kappa = m_evaluate(conf_mat)
        print("val_acc: "+str(val_acc))
        print("val_mean_IoU: "+str(val_mean_IoU))
        print("val_kappa: "+str(val_kappa))
        table = PrettyTable(["序号", "名称", "acc", "IoU"])
        for i in range(8):
            table.add_row([i, class_names[i], val_acc_per_class[i], val_IoU[i]])
        print(table)





if __name__ == "__main__":
    ensemble(['/cache/unet_predict', '/cache/unet_predict1', '/cache/unet_predict2'], r'/cache/result', '.png')




