#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 12:52:56 2022

@author: nmei
"""
import os,torch
from torch import nn,optim
import numpy as np

from utils_deep import (hidden_activation_functions,
                        dataloader,
                        simple_augmentations,
                        vae_train_valid
                        )
from models import (VanillaVAE,
                    simple_classifier
                    )


if __name__ == "__main__":
    dataset_name    = 'CIFAR100'
    experiment_name = 'vanilla_vae'
    model_dir       = os.path.join('../models',experiment_name)
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    f_name          = os.path.join(model_dir,'simple_classifier.h5')
    # image setting
    batch_size          = 64
    image_resize        = 32
    noise_level_train   = 0.
    noise_level_test    = 0.
    rotation            = True
    gitter_color        = False
    # set up random seeds and GPU/CPU
    torch.manual_seed(12345)
    np.random.seed(12345)
    if torch.cuda.is_available():torch.cuda.empty_cache()
    torch.cuda.manual_seed(12345)
    torch.cuda.manual_seed_all(12345)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # training settings
    n_epochs                = int(1e3)
    warmup_epochs           = 3
    patience                = 10
    tol                     = 1e-4
    print_train             = True
    # model settings
    pretrained_model_name   = 'vgg19'
    hidden_units            = 300
    hidden_func_name        = 'leaky_relu'
    hidden_activation       = hidden_activation_functions(hidden_func_name)
    hidden_dropout          = 0.
    latent_units            = 256
    latent_func_name        = 'leaky_relu'
    latent_activation       = hidden_activation_functions(latent_func_name)
    latent_dropout          = 0.
    hidden_dims             = [256,128,64,32]
    model_args              = dict(pretrained_model_name    = pretrained_model_name,
                                   hidden_units             = hidden_units,
                                   hidden_activation        = hidden_activation,
                                   hidden_dropout           = hidden_dropout,
                                   latent_units             = latent_units,
                                   latent_activation        = latent_activation,
                                   latent_dropout           = latent_dropout,
                                   hidden_dims              = [256,128,64,32,16],# as long as we have 5 layers
                                   in_channels              = 3,
                                   in_shape                 = [1,3,image_resize,image_resize],
                                   device                   = device,
                                   )
    # trainer settings
    learning_rate       = 1e-4
    l2_regularization   = 1e-16
    print_train         = True
    # make transforms
    transform                           = simple_augmentations(
                                                image_resize    = image_resize,
                                                noise_level     = noise_level_train,
                                                rotation        = rotation,
                                                gitter_color    = gitter_color,
                                                )
    dataloader_train,dataloader_valid   = dataloader(
                                                dataset_name        = dataset_name,
                                                train               = True,
                                                transform           = transform,
                                                train_valid_split   = [45000,5000],
                                                batch_size          = batch_size,
                                                shuffle             = True,
                                                )
    dataloader_test,_                   = dataloader(
                                                dataset_name    = dataset_name,
                                                train           = False,
                                                transform       = transform,
                                                batch_size      = batch_size,
                                                shuffle         = True,
                                                )
    
    # build the variational autoencoder
    vae             = VanillaVAE(**model_args).to(device)
    # freeze the vae
    for p in vae.parameters(): p.requires_gard = False
    # build the simple classifier
    classifier      = simple_classifier(vae.encoder,
                                        hidden_units,
                                        ).to(device)
    for p in classifier.feature_extractor.parameters():
        p.requires_grad = False
    params          = [p for p in classifier.parameters() if p.requires_grad == True]
    recon_loss_func = nn.BCELoss()
    optimizer       = optim.Adam(params,
                                 lr             = learning_rate,
                                 weight_decay   = l2_regularization,
                                 )
    