#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 00:50:23 2022

@author: vsevolod
"""

import random
import os
import cv2
from matplotlib import pyplot as plt

import albumentations as A

def visualize(image, mask, original_image, original_mask, ppath, nname):
    fontsize = 14
    
    if original_image is None and original_mask is None:
        f, ax = plt.subplots(2, 1, figsize=(8, 7))

        ax[0].imshow(image)
        ax[1].imshow(mask)
    else:
        f, ax = plt.subplots(figsize=(8, 3))
        ax.imshow(original_image)
        ax.set_xticks([])
        ax.set_yticks([])
        f.savefig(os.path.join(ppath, nname+'_original_image.png'))
        
        f, ax = plt.subplots(figsize=(8, 3))
        ax.imshow(original_mask)
        ax.set_xticks([])
        ax.set_yticks([])
        f.savefig(os.path.join(ppath, nname+'_original_mask.png'))

        f, ax = plt.subplots(figsize=(8, 3))
        ax.imshow(image)
        ax.set_xticks([])
        ax.set_yticks([])
        f.savefig(os.path.join(ppath, nname+'_image.png'))

        f, ax = plt.subplots(figsize=(8, 3))
        ax.imshow(mask)
        ax.set_xticks([])
        ax.set_yticks([])
        f.savefig(os.path.join(ppath, nname+'_mask.png'))
        
        
        
        f, ax = plt.subplots(2, 2, figsize=(8, 3))

        ax[0, 0].imshow(original_image)
        ax[0, 0].set_title('Исходное изображение', fontsize=fontsize)
        ax[0, 0].set_xticks([])
        ax[0, 0].set_yticks([])
        
        ax[1, 0].imshow(original_mask)
        ax[1, 0].set_title('Исходная маска', fontsize=fontsize)
        ax[1, 0].set_xticks([])
        ax[1, 0].set_yticks([])
        
        ax[0, 1].imshow(image)
        ax[0, 1].set_title('Преобразованное изображение', fontsize=fontsize)
        ax[0, 1].set_xticks([])
        ax[0, 1].set_yticks([])
        
        ax[1, 1].imshow(mask)
        ax[1, 1].set_title('Преобразованная маска', fontsize=fontsize)
        ax[1, 1].set_xticks([])
        ax[1, 1].set_yticks([])
        

image = cv2.imread('/Path/data/imgs/00182.png')
mask = cv2.imread('/Path/data/masks/00182.png')

original_height, original_width = image.shape[:2]

aug = A.Compose([
    #A.PadIfNeeded(min_height=240, min_width=835, p=0.5, value = 0),
    A.VerticalFlip(p=0.5),              
    A.OneOf([
        A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5),
        A.GridDistortion(p=0.5),
        A.OpticalDistortion(distort_limit=2, shift_limit=0.5, p=1)                  
        ], p=0.8),
    #A.RandomBrightnessContrast(p=0.8, brightness_limit = [-0.1, 0.1], 
                               #contrast_limit = [-0.2, 0.2]),    
    A.RandomGamma(p=0.8)]) 

#random.seed(11)
augmented = aug(image=image, mask=mask)

image_heavy = augmented['image']
mask_heavy = augmented['mask']

visualize(image_heavy, mask_heavy, image, mask, '/Path/graphs/aug', '00182')