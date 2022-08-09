#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 26 12:45:03 2022

@author: darya
"""
import numpy as np
from PIL import Image
import os

def dice_coef(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    smooth = 0.0001
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)


def dice_coef_multilabel(y_true, y_pred, numLabels):
    dice=0
    for index in range(numLabels):
        dice += dice_coef(y_true[:,:,:,index], y_pred[:,:,:,index])
    return dice/numLabels

pathTrueMask = '/Path/data/test/masks/'
pathPredict = '/Path/data/test/predict/'


if __name__ == '__main__':
    num_class = 1
    for i in os.walk(pathTrueMask):
        if len(['.png' in jj.lower() for jj in i[2]]) != 0 and all(['.png' in jj.lower() for jj in i[2]]):
            
            height, width = 480, 1670
            trueMaskArr = np.zeros((len(i[2]), height, width, num_class))
            predMaskArr = np.zeros((len(i[2]), height, width, num_class))
            
            for num, ii in enumerate(i[2]):
                trueMask = np.asarray(Image.open(os.path.join(pathTrueMask, ii)))
                predictMask = np.asarray(Image.open(os.path.join(pathPredict, ii)))
                
                trueMask = np.where(trueMask <= 0.5, 0, 1)
                predictMask = np.where(predictMask <= 0.5, 0, 1)
                
                trueMask = trueMask[..., np.newaxis]
                predictMask = predictMask[..., np.newaxis]
                
                trueMaskArr[num,:,:,:] = trueMask
                predMaskArr[num,:,:,:] = predictMask
                
    resScore = dice_coef_multilabel(y_true = trueMaskArr, y_pred=predMaskArr, numLabels=num_class)