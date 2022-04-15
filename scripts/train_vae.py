#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 13:00:57 2022

@author: nmei
"""
import os,torch
from torch import nn,optim
import numpy as np

from utils_deep import (hidden_activation_functions,
                        dataloader,
                        simple_augmentations,
                        vae_train_valid,
                        clf_train_valid
                        )
from models import (VanillaVAE,
                    simple_classifier)


if __name__ == "__main__":
    dataset_name    = 'CIFAR100'
    experiment_name = 'vanilla_vae'
    model_dir       = os.path.join('../models',experiment_name)
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    f_name          = os.path.join(model_dir,'vae.h5')
    # image setting
    batch_size          = 100
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
    # model settings
    pretrained_model_name   = 'vgg19'
    hidden_units            = 256
    hidden_func_name        = 'selu'
    hidden_activation       = hidden_activation_functions(hidden_func_name)
    hidden_dropout          = 0.
    latent_dropout          = 0.
    hidden_dims             = [256,128,64,32,16]# as long as we have 5 layers
    vae_model_args          = dict(pretrained_model_name    = pretrained_model_name,
                                   hidden_units             = hidden_units,
                                   hidden_activation        = hidden_activation,
                                   hidden_dropout           = hidden_dropout,
                                   hidden_dims              = hidden_dims,
                                   in_channels              = 3,
                                   in_shape                 = [1,3,image_resize,image_resize],
                                   device                   = device,
                                   )
    clf_model_args          = dict(pretrained_model_name    = pretrained_model_name,
                                   # should be the same as the mu and log_var variables
                                   hidden_units             = hidden_units,
                                   ###########################################
                                   hidden_activation        = hidden_activation,
                                   hidden_dropout           = hidden_dropout,
                                   output_units             = 10,
                                   output_activation        = nn.Softmax(dim = -1),
                                   in_shape                 = [1,3,image_resize,image_resize],
                                   device                   = device,
                                   )
    
    # train settings
    learning_rate       = 1e-4
    l2_regularization   = 1e-16
    print_train         = True
    n_epochs            = int(1e3)
    warmup_epochs       = 3
    patience            = 10
    tol                 = 1e-4
    n_noise             = 2
    retrain             = True
    train_args          = dict(device          = device,
                               n_epochs        = n_epochs,
                               print_train     = print_train,
                               warmup_epochs   = warmup_epochs,
                               tol             = tol,
                               patience        = patience,
                               )
    # make transforms
    transform                           = simple_augmentations(
                                                image_resize    = image_resize,
                                                noise_level     = noise_level_train,
                                                rotation        = rotation,
                                                gitter_color    = gitter_color,
                                                )
    dataloader_train,dataloader_valid   = dataloader(
                                                dataset_name        = dataset_name,
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
    vae             = VanillaVAE(**vae_model_args).to(device)
    params          = [p for p in vae.parameters() if p.requires_grad == True]
    recon_loss_func = nn.MSELoss()
    optimizer       = optim.Adam(params,
                                 lr             = learning_rate,
                                 weight_decay   = l2_regularization,
                                 )
    # train the VAE
    if not os.path.exists(f_name) or retrain:
        vae,losses      = vae_train_valid(
                                vae,
                                dataloader_train,
                                dataloader_valid,
                                optimizer,
                                recon_loss_func = recon_loss_func,
                                f_name          = f_name,
                                **train_args
                                )
    else:
        vae.load_state_dict(torch.load(f_name,map_location = device))
    # freeze the vae
    for p in vae.parameters(): p.requires_gard = False
    # build the simple classifier
    classifier      = simple_classifier(**clf_model_args).to(device)
    params          = [p for p in classifier.parameters() if p.requires_grad == True]
    image_loss_func = nn.BCELoss()
    optimizer       = optim.Adam(params,
                                 lr             = learning_rate,
                                 weight_decay   = l2_regularization,
                                 )
    # train the classifier
    classifier,clf_losses = clf_train_valid(
                            classifier,
                            dataloader_train,
                            dataloader_valid,
                            optimizer,
                            image_loss_func = image_loss_func,
                            f_name = f_name.replace('vae.h5','classifier.h5'),
                            n_noise = n_noise,
                            **train_args
                            )
    