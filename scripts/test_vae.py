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
    dataset_name            = 'CIFAR100'
    experiment_name         = 'vanilla_vae'
    train_root              = '../data'
    valid_root              = '../data'
    test_root               = '../data'
    model_dir               = os.path.join('../models',experiment_name)
    f_name                  = os.path.join(model_dir,'vae.h5')
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    # image setting
    batch_size              = 100 # batch size for each epoch
    image_resize            = 32 # image hight
    noise_level_train       = 0. # noise level in training
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
    hidden_units            = 300 # hidden layer units
    hidden_func_name        = 'selu' # hidden layer activation function
    hidden_activation       = hidden_activation_functions(hidden_func_name)
    hidden_dropout          = 0. # hidden layer dropout rate
    hidden_dims             = [300,150,75,30,15]# as long as we have 5 layers
    retrain_encoder         = True # retrain the CNN backbone convolutional layers
    # train settings
    learning_rate           = 1e-3 # initial learning rate, will be reduced by 10 after warmup epochs
    l2_regularization       = 1e-16 # L2 regularization term, used as weight decay
    print_train             = True # print the progresses
    n_epochs                = int(1e3) # max number of epochs
    warmup_epochs           = 5 # we don't save the models in these epochs
    patience                = 10 # we wait for a number of epochs after the best performance
    tol                     = 1e-4 # the difference between the current best and the next best
    n_noise                 = 2 # number of noisy images used in training the classifier
    retrain                 = True # retrain the VAE
    
    vae_model_args          = dict(pretrained_model_name    = pretrained_model_name,
                                   hidden_units             = hidden_units,
                                   hidden_activation        = hidden_activation,
                                   hidden_dropout           = hidden_dropout,
                                   hidden_dims              = hidden_dims,
                                   in_channels              = 3,
                                   in_shape                 = [1,3,image_resize,image_resize],
                                   device                   = device,
                                   retrain_encoder          = retrain_encoder,
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
    train_args              = dict(device          = device,
                                   n_epochs        = n_epochs,
                                   print_train     = print_train,
                                   warmup_epochs   = warmup_epochs,
                                   tol             = tol,
                                   patience        = patience,
                                   )
    # testing settings
    n_noise_levels  = 20
    max_noise_level = np.log10(100)
    noise_levels    = np.concatenate([[0],np.logspace(-1,max_noise_level,n_noise_levels)])
    # build the variational autoencoder
    vae             = VanillaVAE(**vae_model_args).to(device)
    vae.load_state_dict(torch.load(f_name,map_location = device))
    # freeze the vae
    for p in vae.parameters(): p.requires_gard = False
    # build the simple classifier
    classifier      = simple_classifier(**clf_model_args).to(device)
    classifier.load_state_dict(torch.load(f_name.replace('vae.h5','classifier.h5'),map_location = device))
    # freeze the classifier
    for p in classifier.parameters(): p.requires_grad = False
    #
    with torch.no_grad():
        for ii,noise_level in enumerate(noise_levels):
            # make transforms
            transform                           = simple_augmentations(
                                                        image_resize    = image_resize,
                                                        noise_level     = noise_level,
                                                        rotation        = rotation,
                                                        gitter_color    = gitter_color,
                                                        )
            dataloader_test,_                   = dataloader(
                                                        dataset_name    = dataset_name,
                                                        train           = False,
                                                        transform       = transform,
                                                        batch_size      = batch_size,
                                                        shuffle         = True,
                                                        )
            for batch_features,batch_labels in dataloader_test:
                (reconstruction,
                _,
                z,mu,log_var) = vae(batch_features.to(device))
                (hidden_representation,
                 batch_predictions) = classifier(batch_features.to(device))
                asdf