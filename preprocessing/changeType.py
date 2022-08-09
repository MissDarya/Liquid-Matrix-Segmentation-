#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 20:38:15 2022

@author: darya
"""


from PIL import Image
import glob
import os

out_dir = ''
cnt = 0
for img in glob.glob('path/to/images/*.bmp'):
    Image.open(img).resize((300,300)).save(os.path.join(out_dir, str(cnt) + '.png'))
    cnt += 1