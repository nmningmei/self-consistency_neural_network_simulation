#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 13:00:57 2022

@author: nmei
"""
import os,torch
from torch import nn,optim
import numpy as np
from typing import List,Union
from utils_deep import (hidden_activation_functions,
                        dataloader,
                        optimizer_and_scheduler,
                        simple_augmentations,
                        clf_vae_train_valid
                        )
from models import (vae_classifier)


    
if __name__ == "__main__":
    dataset_name            = None
    experiment_name         = 'clf+vae'
    train_root              = '../data/greyscaled'
    valid_root              = '../data/experiment_images_greyscaled'
    test_root               = '../data/metasema_images'
    model_dir               = os.path.join('../models',experiment_name)
    figure_dir              = os.path.join('../figures',experiment_name)
    f_name                  = os.path.join(model_dir,'clf-vae.h5')
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    if not os.path.exists(figure_dir):
        os.mkdir(figure_dir)
    # image setting
    batch_size              = 16 # batch size for each epoch
    image_resize            = 128 # image hight
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
    latent_func_names       = ['tanh','tanh'] # mu and log_var layer activation function
    latent_activations      = [hidden_activation_functions(item) for item in latent_func_names]
    hidden_dropout          = 0. # hidden layer dropout rate
    hidden_dims             = [hidden_units,
                               int(hidden_units/2),
                               int(hidden_units/4),
                               int(hidden_units/8),
                               int(hidden_units/16),
                               ]
    output_units            = 2
    vae_out_func_name       = hidden_func_name#'relu' # activation function of the reconstruction
    vae_output_activation   = hidden_activation_functions(vae_out_func_name)
    retrain_encoder         = False # retrain the CNN backbone convolutional layers
    multi_hidden_layer      = True # add more dense layer to the classifier and the VAE encoder
    if multi_hidden_layer:
        f_name              = f_name.replace('.h5','_deep.h5')
    latent_units            = hidden_dims[-1] if multi_hidden_layer else int(hidden_units/2)
    # train settings
    learning_rate           = 1e-4 # initial learning rate, will be reduced by 10 after warmup epochs
    l2_regularization       = 1e-4 # L2 regularization term, used as weight decay
    print_train             = True # print the progresses
    n_epochs                = int(1e3) # max number of epochs
    warmup_epochs           = 10 # we don't save the models in these epochs
    patience                = 20 # we wait for a number of epochs after the best performance
    tol                     = 1e-4 # the difference between the current best and the next best
    n_noise                 = int(batch_size/8) # number of noisy images used in training the classifier
    retrain                 = True # retrain the VAE
    # testing settings
    n_noise_levels          = 50
    max_noise_level         = np.log10(1000)
    noise_levels            = np.concatenate([[0],np.logspace(-2,max_noise_level,n_noise_levels)])
    
    model_args          = dict(pretrained_model_name    = pretrained_model_name,
                               hidden_units             = hidden_units,
                               hidden_activation        = hidden_activation,
                               hidden_dropout           = hidden_dropout,
                               hidden_dims              = hidden_dims,
                               output_units             = output_units,
                               latent_units             = latent_units,
                               vae_output_activation    = vae_output_activation,
                               latent_activations       = latent_activations,
                               retrain_encoder          = retrain_encoder,
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
                                   factor               = .5,# factor of reducing the learning rate for the scheduler
                                   patience             = 10,
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
    dataloader_train,_                  = dataloader(
                                                dataset_name    = dataset_name,
                                                root            = train_root,
                                                train           = True,
                                                transform       = transform,
                                                batch_size      = batch_size,
                                                shuffle         = True,
                                                )
    dataloader_valid,_                  = dataloader(
                                                dataset_name    = dataset_name,
                                                root            = valid_root,
                                                train           = True,
                                                transform       = transform,
                                                batch_size      = batch_size,
                                                shuffle         = True,
                                                )
    # dataloader_train,dataloader_valid   = dataloader(
    #                                             dataset_name        = dataset_name,
    #                                             transform           = transform,
    #                                             train_valid_split   = [45000,5000],
    #                                             batch_size          = batch_size,
    #                                             shuffle             = True,
    #                                             )
    dataloader_test,_                   = dataloader(
                                                dataset_name    = dataset_name,
                                                root            = test_root,
                                                train           = False,
                                                transform       = transform,
                                                batch_size      = batch_size,
                                                shuffle         = True,
                                                )
    
    ###########################################################################
    # build the variational autoencoder
    print('Build CLF-VAE model')
    vae             = vae_classifier(**model_args).to(device)
    # CNN + hidden_layer + output_layer
    if retrain_encoder:
        params_clf  = [p for p in vae.encoder.parameters()]
    else:
        params_clf  = []
    for p in vae.hidden_layer.parameters():params_clf.append(p)
    for p in vae.output_layer.parameters():params_clf.append(p)
    params_clf      = [p for p in params_clf if p.requires_grad == True]
    # mu + log_var + decoder
    params_vae      = [p for p in vae.mu_layer.parameters()]
    for p in vae.log_var_layer.parameters():params_vae.append(p)
    for p in vae.decoder.parameters():params_vae.append(p)
    paras_vae       = [p for p in params_vae if p.requires_grad == True]
    recon_loss_func = nn.CosineEmbeddingLoss()
    image_loss_func = nn.BCELoss()
    (optimizer1,
     scheduler1)    = optimizer_and_scheduler(params = params_clf,**optim_args)
    (optimizer2,
     scheduler2)    = optimizer_and_scheduler(params = params_vae,**optim_args)
    
    # train the VAE
    if not os.path.exists(f_name) or retrain:
        print('Train the model')
        vae,losses  = clf_vae_train_valid(
                                vae,
                                dataloader_train,
                                dataloader_valid,
                                [optimizer1,optimizer2],
                                [scheduler1,scheduler2],
                                image_loss_func = image_loss_func,
                                recon_loss_func = recon_loss_func,
                                f_name          = f_name,
                                patience        = patience,
                                beta            = 1.,# since the reconstruction is not ideal, and all we want is the learned sampling distributions, we weight more on the variational loss
                                **train_args
                                )
    else:
        print('Load the model weights')
        vae.load_state_dict(torch.load(f_name,map_location = device))
    # freeze the vae
    for p in vae.parameters(): p.requires_gard = False
    
    