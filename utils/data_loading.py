import logging
from os import listdir
from os.path import splitext
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import albumentations as A


class BasicDataset(Dataset):
    def __init__(self, images_dir: str, masks_dir: str, scale: float = 1.0, mask_suffix: str = ''):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix
        self.aug = A.Compose([
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
        self.ids = [splitext(file)[0] for file in listdir(images_dir) if not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, scale, is_mask):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img_ndarray = np.asarray(pil_img)

        #if img_ndarray.ndim == 2 and not is_mask:
            #img_ndarray = img_ndarray[np.newaxis, ...]
            
        #if img_ndarray.ndim == 2:
            #img_ndarray = img_ndarray[np.newaxis, ...]
        #elif not is_mask:
            #img_ndarray = img_ndarray.transpose((2, 0, 1))

        if not is_mask:
            img_ndarray = img_ndarray / 255

        return img_ndarray
      
    @classmethod
    def preprocessPredict (cls, pil_img, scale, is_mask):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img_ndarray = np.asarray(pil_img)

        if img_ndarray.ndim == 2 and not is_mask:
            img_ndarray = img_ndarray[np.newaxis, ...]
            
        elif not is_mask:
            img_ndarray = img_ndarray.transpose((2, 0, 1))

        if not is_mask:
            img_ndarray = img_ndarray / 255

        return img_ndarray

    @classmethod
    def load(cls, filename):
        ext = splitext(filename)[1]
        if ext in ['.npz', '.npy']:
            return Image.fromarray(np.load(filename))
        elif ext in ['.pt', '.pth']:
            return Image.fromarray(torch.load(filename).numpy())
        else:
            return Image.open(filename)

    def __getitem__(self, idx):
        name = self.ids[idx]

        mask_file = list(self.masks_dir.glob(name + self.mask_suffix + '.*'))
        
        img_file = list(self.images_dir.glob(name + '.*'))

        assert len(mask_file) == 1, 'Either no mask or multiple masks found for the ID {0}'.format(name + self.mask_suffix)
#assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        mask = self.load(mask_file[0])
        img = self.load(img_file[0])

        #print(mask.size)

        # print(img.size)

        assert img.size == mask.size, \
            'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(img, self.scale, is_mask=False)
        mask = self.preprocess(mask, self.scale, is_mask=True)
        
        # print(type(mask))
        
        #original_height, original_width = img.shape[1:]

        
        #img = np.transpose(np.array(img), axes=[1,2,0])
        #mask = np.transpose(np.array(mask), axes=[1,2,0])

        #random.seed(11)
        augmented = self.aug(image=img, mask=mask)

        img = augmented['image']
        mask = augmented['mask']
        
        #img = np.transpose(np.array(img), axes=[2,0,1])
        #mask = np.transpose(np.array(mask), axes=[2,0,1])
        
        img = img[np.newaxis, ...]
        #mask = mask[np.newaxis, ...]
        
        # print('img')
        # print(img.shape)
        
        # print('mask')
        # print(mask.shape)
        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }
