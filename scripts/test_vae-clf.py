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
                        optimizer_and_scheduler,
                        simple_augmentations,
                        clf_vae_train_valid
                        )
from models import (vae_classifier)


    
if __name__ == "__main__":
    dataset_name            = 'CIFAR100'
    experiment_name         = 'vanilla_vae'
    train_root              = '../data'
    valid_root              = '../data'
    test_root               = '../data'
    model_dir               = os.path.join('../models',experiment_name)
    f_name                  = os.path.join(model_dir,'clf-vae.h5')
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    # image setting
    batch_size              = 32 # batch size for each epoch
    image_resize            = 32 # image hight
    noise_level_train       = 1e-3 # noise level in training
    noise_level_test        = 0. # noise level in testing
    rotation                = True # image augmentation
    gitter_color            = False # image augmentation for Gabor patches
    # set up random seeds and GPU/CPU
    torch.manual_seed(12345)
    np.random.seed(12345)
    if torch.cuda.is_available():torch.cuda.empty_cache()
    torch.cuda.manual_seed(12345)
    torch.cuda.manual_seed_all(12345)
    device                  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model settings
    pretrained_model_name   = 'vgg19'
    hidden_units            = 256 # hidden layer units
    hidden_func_name        = 'relu' # hidden layer activation function
    hidden_activation       = hidden_activation_functions(hidden_func_name)
    latent_func_name        = 'leaky_relu'
    latent_activation       = hidden_activation_functions(latent_func_name)
    hidden_dropout          = 0. # hidden layer dropout rate
    hidden_dims             = [hidden_units,
                               int(hidden_units/2),
                               int(hidden_units/4),
                               int(hidden_units/8),
                               # int(hidden_units/16),
                               ]# as long as we have 5 layers
    vae_out_func_name       = 'tanh'
    vae_output_activation   = hidden_activation_functions(vae_out_func_name)
    retrain_encoder         = True # retrain the CNN backbone convolutional layers
    multi_hidden_layer      = False # add more dense layer to the classifier and the VAE encoder
    # train settings
    learning_rate           = 1e-4 # initial learning rate, will be reduced by 10 after warmup epochs
    l2_regularization       = 1e-16 # L2 regularization term, used as weight decay
    print_train             = True # print the progresses
    n_epochs                = int(1e3) # max number of epochs
    warmup_epochs           = 5 # we don't save the models in these epochs
    patience                = 10 # we wait for a number of epochs after the best performance
    tol                     = 1e-4 # the difference between the current best and the next best
    n_noise                 = 0 # number of noisy images used in training the classifier
    retrain                 = True # retrain the VAE
    
    latent_units            = hidden_dims[-1] if multi_hidden_layer else hidden_units
    model_args          = dict(pretrained_model_name    = pretrained_model_name,
                               hidden_units             = hidden_units,
                               hidden_activation        = hidden_activation,
                               hidden_dropout           = hidden_dropout,
                               hidden_dims              = hidden_dims,
                               latent_units             = hidden_dims[-1],
                               vae_output_activation    = vae_output_activation,
                               latent_activation        = latent_activation,
                               in_channels              = 3,
                               in_shape                 = [1,3,image_resize,image_resize],
                               device                   = device,
                               multi_hidden_layer       = multi_hidden_layer,
                               clf_output_activation    = nn.Softmax(dim = -1),
                               )
    
    train_args              = dict(device          = device,
                                   n_epochs        = n_epochs,
                                   print_train     = print_train,
                                   warmup_epochs   = warmup_epochs,
                                   tol             = tol,
                                   # patience        = patience,
                                   )
    optim_args              = dict(learning_rate        = learning_rate,
                                   l2_regularization    = l2_regularization,
                                   mode                 = 'min',
                                   factor               = .5,
                                   patience             = int(patience/2),
                                   threshold            = tol,
                                   min_lr               = 1e-8,
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
    
    ###########################################################################
    # build the variational autoencoder
    print('Build CLF-VAE model')
    vae             = vae_classifier(**model_args).to(device)
    for p in vae.parameters():p.requires_grad == False
    
    