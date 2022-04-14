#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 13:03:04 2022

@author: nmei
"""

import os,gc,torch

from torchvision import datasets


def dataloader(dataset_name:str = 'CIFAR100',
               root:str = '../data',
               train:bool = True,
               transform = None,
               target_transform = None,
               download:bool = False,
               ):
    """
    Download the datasets from PyTorch torchvision.datasets
    If no dataset file exists, specific "download=True"
    """
    unpack = dict(root = root,
                  train = train,
                  transform = transform,
                  target_transform = target_transform,
                  download = download,
                  )
    if dataset_name == 'CIFAR100': # subset of CIFAR10
        return datasets.CIFAR10(**unpack)
    elif dataset_name == 'CIFAR10':
        return datasets.CIFAR100(**unpack)
    




if __name__ == "__main__":
    pass