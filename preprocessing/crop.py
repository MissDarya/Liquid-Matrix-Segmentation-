#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 20:41:02 2022

@author: darya
"""


from PIL import Image
import os.path, sys

path = 'YourPath'
dirs = os.listdir(path)

def crop():
    for item in dirs:
        fullpath = os.path.join(path,item)         #corrected
        if os.path.isfile(fullpath):
            im = Image.open(fullpath)
            f, e = os.path.splitext(fullpath)
            imCrop = im.crop((30, 10, 1024, 1004)) #corrected
            imCrop.save(f + 'Cropped.bmp', "BMP", quality=100)

crop()