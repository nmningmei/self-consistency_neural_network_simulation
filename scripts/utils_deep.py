#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 13:03:04 2022

@author: nmei
"""
from typing import List, Callable, Union, Any, TypeVar, Tuple, List, Optional
import os,gc
from glob        import glob
from tqdm        import tqdm
from collections import OrderedDict
import pandas as pd
import numpy  as np

import torch
from torch          import nn,no_grad
from torch.utils    import data
from torch.nn       import functional as F
from torch          import optim
from torch.autograd import Variable

from torchvision.datasets import ImageFolder
from torchvision          import transforms,datasets
from torchvision          import models as Tmodels

from sklearn                 import metrics
from sklearn.preprocessing   import StandardScaler
from sklearn.svm             import LinearSVC,SVC
from sklearn.decomposition   import PCA
from sklearn.pipeline        import make_pipeline
from sklearn.model_selection import StratifiedShuffleSplit,cross_validate,permutation_test_score
from sklearn.calibration     import CalibratedClassifierCV
from sklearn.linear_model    import LogisticRegression
from sklearn.utils           import shuffle as sk_shuffle
from sklearn.ensemble        import RandomForestClassifier
from sklearn.metrics         import roc_auc_score

def standard_dataset(generator,
                     train_valid_split,
                     train,
                     unpack,
                     ) -> Tuple:
    if train_valid_split is not None:
        train,valid = data.random_split(generator,train_valid_split,)
        loader_train = data.DataLoader(train,**unpack)
        loader_valid = data.DataLoader(valid,**unpack)
        return loader_train,loader_valid
    elif train_valid_split == None:
        loader_test = data.DataLoader(generator,**unpack)
        return loader_test,None
    else:
        raise NotImplementedError

def dataloader(dataset_name:str = 'CIFAR10',
               root:str = '../data',
               train:bool = True,
               transform:Optional[Callable] = None,
               target_transform:Optional[Callable] = None,
               download:bool = False,
               train_valid_split:Optional[List] = None,
               batch_size:int              = 8,
               num_workers:int             = 2,
               shuffle:bool                = True,
               return_path:bool            = False,
               ):
    """
    Download the datasets from PyTorch torchvision.datasets
    If no dataset file exists, specific "download=True"
    """
    unpack1 = dict(root = root,
                   train = train,
                   transform = transform,
                   target_transform = target_transform,
                   download = download,
                   )
    unpack2 = dict(batch_size = batch_size,
                   num_workers = num_workers,
                   shuffle = shuffle,
                   )
    if dataset_name == 'CIFAR100': # subset of CIFAR10
        generator = datasets.CIFAR10(**unpack1)
        loaders = standard_dataset(generator, 
                                   train_valid_split,
                                   train,
                                   unpack2)
    elif dataset_name == 'CIFAR10':
        generator = datasets.CIFAR100(**unpack1)
        loaders = standard_dataset(generator, 
                                   train_valid_split,
                                   train,
                                   unpack2)
    elif dataset_name == None:
        loader = data_loader(data_root = root,
                             augmentations = transform,
                             return_path = False,
                             **unpack2,)
        loaders = loader,None
    else:
        raise NotImplementedError
    return loaders

class customizedDataset(ImageFolder):
    def __getitem__(self, idx):
        original_tuple  = super(customizedDataset,self).__getitem__(idx)
        path = self.imgs[idx][0]
        tuple_with_path = (original_tuple +  (path,))
        return tuple_with_path

def data_loader(data_root:str,
                augmentations:transforms    = None,
                batch_size:int              = 8,
                num_workers:int             = 2,
                shuffle:bool                = True,
                return_path:bool            = False,
                )->data.DataLoader:
    """
    Create a batch data loader from a given image folder.
    The folder must be organized as follows:
        main ---
             |
             -----class 1 ---
                         |
                         ----- image 1.jpeg
                         .
                         .
                         .
            |
            -----class 2 ---
                        |
                        ---- image 1.jpeg
                        .
                        .
                        .
            |
            -----class 3 ---
                        |
                        ---- image 1.jpeg
                        .
                        .
                        .
    Input
    --------------------------
    data_root: str, the main folder
    augmentations: torchvision.transformers.Compose, steps of augmentation
    batch_size: int, batch size
    num_workers: int, CPU --> GPU carrier, number of CPUs
    shuffle: Boolean, whether to shuffle the order
    return_pth: Boolean, lod the image paths

    Output
    --------------------------
    loader: DataLoader, a Pytorch dataloader object
    """
    if return_path:
        datasets = customizedDataset(
                root                        = data_root,
                transform                   = augmentations
                )
    else:
        datasets    = ImageFolder(
                root                        = data_root,
                transform                   = augmentations
                )
    loader      = data.DataLoader(
                datasets,
                batch_size                  = batch_size,
                num_workers                 = num_workers,
                shuffle                     = shuffle,
                )
    return loader

#candidate models
def candidates(model_name,pretrained = True,):
    picked_models = dict(
            resnet18        = Tmodels.resnet18(pretrained           = pretrained,
                                              progress              = False,),
            alexnet         = Tmodels.alexnet(pretrained            = pretrained,
                                             progress               = False,),
            # squeezenet      = Tmodels.squeezenet1_1(pretrained      = pretrained,
            #                                        progress         = False,),
            vgg19           = Tmodels.vgg19_bn(pretrained           = pretrained,
                                              progress              = False,),
            densenet169     = Tmodels.densenet169(pretrained        = pretrained,
                                                 progress           = False,),
            inception       = Tmodels.inception_v3(pretrained       = pretrained,
                                                  progress          = False,),
            # googlenet       = Tmodels.googlenet(pretrained          = pretrained,
            #                                    progress             = False,),
            # shufflenet      = Tmodels.shufflenet_v2_x0_5(pretrained = pretrained,
            #                                             progress    = False,),
            mobilenet       = Tmodels.mobilenet_v2(pretrained       = pretrained,
                                                  progress          = False,),
            # resnext50_32x4d = Tmodels.resnext50_32x4d(pretrained    = pretrained,
            #                                          progress       = False,),
            resnet50        = Tmodels.resnet50(pretrained           = pretrained,
                                              progress              = False,),
            )
    return picked_models[model_name]

def define_type(model_name):
    model_type          = dict(
            alexnet     = 'simple',
            vgg19       = 'simple',
            densenet169 = 'simple',
            inception   = 'inception',
            mobilenet   = 'simple',
            resnet18    = 'resnet',
            resnet50    = 'resnet',
            )
    return model_type[model_name]

def hidden_activation_functions(activation_func_name):
    funcs = dict(relu = nn.ReLU(),
                 selu = nn.SELU(),
                 elu = nn.ELU(),
                 sigmoid = nn.Sigmoid(),
                 tanh = nn.Tanh(),
                 linear = None,
                 leaky_relu = nn.LeakyReLU(),
                 )
    return funcs[activation_func_name]

def noise_fuc(x,noise_level = 1):
    """
    add guassian noise to the images during agumentation procedures

    Inputs
    --------------------
    x: torch.tensor, batch_size x 3 x height x width
    noise_level: float, standard deviation of the gaussian distribution
    """
    generator = torch.distributions.normal.Normal(0,noise_level)
    return x + generator.sample(x.shape)

def simple_augmentations(image_resize   = 128,
                         noise_level    = None,
                         rotation       = True,
                         gitter_color   = False,
                         ):
    """
    Simple augmentation steps
    
    Inputs 
    ---
    image_resize: int, the height and width of the images
    noise_level: float, standard deviation of the Gaussian distribution the noise is sampled from
    rotation: bool, one of the augmentation methods, for object recognition only
    gitter_color: bool, one of the augmentation methods, for Gabor only
    
    Outputs
    ---
    torchvision.transformer object
    """
    steps = [transforms.Resize((image_resize,image_resize)),]
    
    if rotation and not gitter_color:
        steps.append(transforms.RandomHorizontalFlip(p = 0.5))
        steps.append(transforms.RandomRotation(45,))
        steps.append(transforms.RandomVerticalFlip(p = 0.5))
    elif gitter_color and not rotation:
        steps.append(transforms.RandomCrop((image_resize,image_resize)))
        steps.append(transforms.ColorJitter(brightness = 0.25,
                                            contrast = 0.25,
                                            saturation = 0.25,
                                            hue = 0.25,))
    
    steps.append(transforms.ToTensor())
    if noise_level > 0:
        steps.append(transforms.Lambda(lambda x:noise_fuc(x,noise_level)))
    steps.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    # mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    transform_steps = transforms.Compose(steps)
    return transform_steps


if __name__ == "__main__":
    pass