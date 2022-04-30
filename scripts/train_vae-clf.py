#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 13:00:57 2022

@author: nmei
"""
import os,torch
from torch import nn
import numpy as np
from utils_deep import (dataloader,
                        optimizer_and_scheduler,
                        simple_augmentations,
                        clf_vae_train_valid
                        )
from models import (vae_classifier)
import experiment_settings

if __name__ == "__main__":
    # set up random seeds and GPU/CPU
    torch.manual_seed(12345)
    np.random.seed(12345)
    if torch.cuda.is_available():torch.cuda.empty_cache()
    torch.cuda.manual_seed(12345)
    torch.cuda.manual_seed_all(12345)
    device                  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for d in [experiment_settings.model_dir,
              experiment_settings.figure_dir,
              experiment_settings.results_dir]:
        if not os.path.exists(d):
            os.mkdir(d)
    
    model_args          = dict(pretrained_model_name    = experiment_settings.pretrained_model_name,
                               hidden_units             = experiment_settings.hidden_units,
                               hidden_activation        = experiment_settings.hidden_activation,
                               hidden_dropout           = experiment_settings.hidden_dropout,
                               hidden_dims              = experiment_settings.hidden_dims,
                               output_units             = experiment_settings.output_units,
                               latent_units             = experiment_settings.latent_units,
                               vae_output_activation    = experiment_settings.vae_output_activation,
                               latent_activations       = experiment_settings.latent_activations,
                               retrain_encoder          = experiment_settings.retrain_encoder,
                               in_channels              = 3,
                               in_shape                 = [1,3,
                                                           experiment_settings.image_resize,
                                                           experiment_settings.image_resize],
                               device                   = device,
                               multi_hidden_layer       = experiment_settings.multi_hidden_layer,
                               clf_output_activation    = nn.Softmax(dim = -1),
                               )
    
    train_args              = dict(device           = device,
                                   n_epochs         = experiment_settings.n_epochs,
                                   print_train      = experiment_settings.print_train,
                                   warmup_epochs    = experiment_settings.warmup_epochs,
                                   tol              = experiment_settings.tol,
                                   image_resize     = experiment_settings.image_resize
                                   # patience        = patience,
                                   )
    optim_args              = dict(learning_rate        = experiment_settings.learning_rate,
                                   l2_regularization    = experiment_settings.l2_regularization,
                                   mode                 = 'min',
                                   factor               = .5,# factor of reducing the learning rate for the scheduler
                                   patience             = 10,
                                   threshold            = experiment_settings.tol,
                                   min_lr               = 1e-8,
                                   )
    # make transforms
    transform                           = simple_augmentations(
                                                image_resize    = experiment_settings.image_resize,
                                                noise_level     = experiment_settings.noise_level_train,
                                                rotation        = experiment_settings.rotation,
                                                gitter_color    = experiment_settings.gitter_color,
                                                )
    dataloader_train,_                  = dataloader(
                                                dataset_name    = experiment_settings.dataset_name,
                                                root            = experiment_settings.train_root,
                                                train           = True,
                                                transform       = transform,
                                                batch_size      = experiment_settings.batch_size,
                                                shuffle         = True,
                                                )
    dataloader_valid,_                  = dataloader(
                                                dataset_name    = experiment_settings.dataset_name,
                                                root            = experiment_settings.valid_root,
                                                train           = True,
                                                transform       = transform,
                                                batch_size      = experiment_settings.batch_size,
                                                shuffle         = True,
                                                )
    # dataloader_train,dataloader_valid   = dataloader(
    #                                             dataset_name        = experiment_settings.dataset_name,
    #                                             transform           = transform,
    #                                             train_valid_split   = [45000,5000],
    #                                             batch_size          = experiment_settings.batch_size,
    #                                             shuffle             = True,
    #                                             )
    dataloader_test,_                   = dataloader(
                                                dataset_name    = experiment_settings.dataset_name,
                                                root            = experiment_settings.test_root,
                                                train           = False,
                                                transform       = transform,
                                                batch_size      = experiment_settings.batch_size,
                                                shuffle         = True,
                                                )
    
    ###########################################################################
    # build the variational autoencoder
    print(f'Build CLF-VAE model on CNN backbone {experiment_settings.pretrained_model_name}')
    vae             = vae_classifier(**model_args).to(device)
    # CNN + hidden_layer + output_layer
    if experiment_settings.retrain_encoder:
        params_clf  = [{'params':vae.encoder.parameters(),'lr':1e-8}]
    else:
        params_clf  = []
    params_clf.append({'params':vae.hidden_layer.parameters()})
    params_clf.append({'params':vae.output_layer.parameters()})
    # mu + log_var + decoder
    params_vae      = [p for p in vae.mu_layer.parameters()]
    for p in vae.log_var_layer.parameters():params_vae.append(p)
    for p in vae.decoder.parameters():params_vae.append(p)
    paras_vae       = [p for p in params_vae if p.requires_grad == True]
    recon_loss_func = nn.CosineEmbeddingLoss(margin = 2.,reduction = 'sum')
    image_loss_func = nn.BCELoss()
    (optimizer1,
     scheduler1)    = optimizer_and_scheduler(params = params_clf,**optim_args)
    (optimizer2,
     scheduler2)    = optimizer_and_scheduler(params = params_vae,**optim_args)
    
    # train the VAE
    if not os.path.exists(experiment_settings.f_name) or experiment_settings.retrain:
        print('Train the model')
        vae,losses  = clf_vae_train_valid(
                                vae,
                                dataloader_train,
                                dataloader_valid,
                                [optimizer1,optimizer2],
                                [scheduler1,scheduler2],
                                transform       = simple_augmentations,
                                image_loss_func = image_loss_func,
                                recon_loss_func = recon_loss_func,
                                f_name          = experiment_settings.f_name,
                                patience        = experiment_settings.patience,
                                beta            = 10.,# since the reconstruction is not ideal, and all we want is the learned sampling distributions, we weight more on the variational loss
                                **train_args
                                )
    else:
        print('Load the model weights')
        vae.load_state_dict(torch.load(experiment_settings.f_name,map_location = device))
    # freeze the vae
    for p in vae.parameters(): p.requires_gard = False
    
    