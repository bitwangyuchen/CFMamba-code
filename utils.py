import cv2
import numpy as np
import torch.nn as nn
import os
import pandas as pd
import torch
import torch.nn.functional as F


def csv_to_numpy(file_path):
    df = pd.read_csv(file_path, header=None)

    array = np.array(df.values, dtype=np.float32)
    return array
def sdm_srf_ircut():
    s_ircut=csv_to_numpy('./resources/sdm_input_ircut.csv')
    return s_ircut
    
def sdm_srf_irpass():
    s_irpass=csv_to_numpy('./resources/sdm_input_irpass.csv')
    return s_irpass
def omsiv_srf_ircut():
    s_ircut=csv_to_numpy('./resources/omsiv_input_ircut.csv')
    return s_ircut
    
def omsiv_srf_irpass():
    s_irpass=csv_to_numpy('./resources/omsiv_input_irpass.csv')
    return s_irpass
def awb(image, percentile=0.9):

    img_float = image.astype(np.float32) / 255.0
    

    luminance = 0.299 * img_float[:, :, 2] + 0.587 * img_float[:, :, 1] + 0.114 * img_float[:, :, 0]
    

    threshold = np.percentile(luminance, percentile * 100)
    mask = luminance >= threshold
    

    b_white = np.mean(img_float[:, :, 0][mask])
    g_white = np.mean(img_float[:, :, 1][mask])
    r_white = np.mean(img_float[:, :, 2][mask])
    

    balanced = img_float.copy()
    balanced[:, :, 0] = np.clip(balanced[:, :, 0] / b_white, 0, 1)
    balanced[:, :, 1] = np.clip(balanced[:, :, 1] / g_white, 0, 1)
    balanced[:, :, 2] = np.clip(balanced[:, :, 2] / r_white, 0, 1)
    

    return (balanced * 255).astype(np.uint8)


def cv_imwrite(img,savepath):
    img=awb(img)
    ircut = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(savepath,ircut)

def image_normalization(img, img_min=0, img_max=255):

    img = np.float32(img)
    epsilon=1e-12 
    img = (img-np.min(img))*(img_max-img_min)/((np.max(img)-np.min(img))+epsilon)+img_min
    return img

class MAE(nn.Module):
    def __init__(self):
        super(MAE, self).__init__()

    def forward(self, gt, pred):
        """
        pred : (b,c,w,h)
        gt : (b,c,w,h)
        """
        pred = torch.clamp(pred, 0, 1)
        gt = torch.clamp(gt, 0, 1)
        cos_similarity = F.cosine_similarity(pred+1e-4,gt+1e-4,dim=1)
        cos_similarity = torch.clamp(cos_similarity, -1, 1)
        rad = torch.acos(cos_similarity)
        ang_error = torch.rad2deg(rad)

        mean_angular_error = torch.mean(ang_error.reshape(-1))
        return mean_angular_error

