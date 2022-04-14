#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 13:00:57 2022

@author: nmei
"""
import torch

import numpy as np

from utils_deep import (dataloader,
                        simple_augmentations
                        )



if __name__ == "__main__":
<<<<<<< HEAD
    dataset_name = 'CIFAR100'
=======
    dataset_name = 'CIFAR10'
>>>>>>> 0e761d6c4dff6b05ee175c631df0b0771c541ba7
    # image setting
    image_resize = 32
    noise_level_train = 0.
    noise_level_test = 0.
    rotation = True
    gitter_color = False
    # set up random seeds and GPU/CPU
    torch.manual_seed(12345)
    
    # make transforms
    transform = simple_augmentations(image_resize = image_resize,
                                     noise_level = noise_level_train,
                                     rotation = rotation,
                                     gitter_color = gitter_color,
                                     )
    dataloader_train,dataloader_valid = dataloader(dataset_name = dataset_name,
                                                   train = True,
                                                   transform = transform,
                                                   train_valid_split = [45000,5000],
                                                   )
<<<<<<< HEAD
    dataloader_test,_ = dataloader(dataset_name = dataset_name,
                                   train = False,
=======
    dataloader_test,_ = dataloader(dataset_name = dataset_name,train = False,
>>>>>>> 0e761d6c4dff6b05ee175c631df0b0771c541ba7
                                   transform = transform,
                                   )
    
    # build the variational autoencoder
    




