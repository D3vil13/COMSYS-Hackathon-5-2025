"""
This module defines data augmentation and preprocessing transforms for training and validation using the Albumentations library.
Functions:
    get_train_transforms():
        Constructs and returns a composed Albumentations transform for training images.
        The pipeline includes resizing, flipping, shifting, scaling, rotating, and optional augmentations such as blur, fog, brightness/contrast adjustment, grayscale conversion, and rain effects, based on configuration flags.
        The final steps normalize the image and convert it to a PyTorch tensor.
    get_val_transforms():
        Constructs and returns a composed Albumentations transform for validation images.
        The pipeline includes resizing, normalization, and conversion to a PyTorch tensor.
Classes:
    AlbumentationsTransform:
        Wrapper class for applying an Albumentations transform to a PIL Image.
        Args:
            aug (albumentations.Compose): The composed Albumentations transform to apply.
        Methods:
            __call__(img: Image.Image):
                Applies the transform to a PIL Image and returns the resulting tensor.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
from config import cfg
from PIL import Image
import numpy as np

def get_train_transforms():
    tfms = [A.Resize(cfg.input_size, cfg.input_size)]
    
    tfms += [
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
    A.RandomShadow(p=0.3),
]
 
    if cfg.use_blur:
        tfms.append(A.OneOf([A.MotionBlur(p=0.5), A.GaussianBlur(p=0.5)], p=0.5))
    if cfg.use_fog:
        tfms.append(A.RandomFog(p=0.6))
    if cfg.use_brightness:
        tfms.append(A.RandomBrightnessContrast(p=0.5))
    if cfg.use_greyscale:
        tfms.append(A.ToGray(p=0.3))
    if cfg.use_rain:
        tfms.append(A.RandomRain(p=0.3))
    
    tfms += [A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), ToTensorV2()]
    return A.Compose(tfms)

def get_val_transforms():
    return A.Compose([
        A.Resize(cfg.input_size, cfg.input_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])



class AlbumentationsTransform:
    def __init__(self, aug):
        self.aug = aug

    def __call__(self, img: Image.Image):
        image_np = np.array(img)  # Convert PIL → NumPy
        augmented = self.aug(image=image_np)
        return augmented["image"]  # This is already a tensor from ToTensorV2
