#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 13:00:57 2022

@author: nmei
"""
import os,torch
from torch import nn,optim
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.spatial import distance

from utils_deep import (hidden_activation_functions,
                        dataloader,
                        simple_augmentations,
                        vae_train_valid,
                        clf_train_valid
                        )
from models import (vae_classifier)

from matplotlib import pyplot as plt

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
    if multi_hidden_layer:
        f_name              = f_name.replace('.h5','_deep.h5')
    # train settings
    learning_rate           = 1e-4 # initial learning rate, will be reduced by 10 after warmup epochs
    l2_regularization       = 1e-16 # L2 regularization term, used as weight decay
    print_train             = True # print the progresses
    n_epochs                = int(1e3) # max number of epochs
    warmup_epochs           = 5 # we don't save the models in these epochs
    patience                = 20 # we wait for a number of epochs after the best performance
    tol                     = 1e-4 # the difference between the current best and the next best
    n_noise                 = 0 # number of noisy images used in training the classifier
    retrain                 = True # retrain the VAE
    # testing settings
    n_noise_levels  = 20
    max_noise_level = np.log10(100)
    noise_levels    = np.concatenate([[0],np.logspace(-1,max_noise_level,n_noise_levels)])
    
    latent_units            = hidden_dims[-1] if multi_hidden_layer else hidden_units
    model_args          = dict(pretrained_model_name    = pretrained_model_name,
                               hidden_units             = hidden_units,
                               hidden_activation        = hidden_activation,
                               hidden_dropout           = hidden_dropout,
                               hidden_dims              = hidden_dims,
                               latent_units             = latent_units,
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
    
    # build the variational autoencoder
    print('Build CLF-VAE model')
    vae             = vae_classifier(**model_args).to(device)
    vae.load_state_dict(torch.load(f_name,map_location = device))
    # freeze the vae
    for p in vae.parameters(): p.requires_gard = False
    
    #
    results = dict(noise_level = [],
                   accuracy = [],
                   )
    df_res = dict(noise_level = [],
                  y_true = [],
                  y_pred = [],
                  y_prob = [],
                  )
    with torch.no_grad():
        hidden_representations = []
        sampled_representations = []
        distances = []
        for ii,noise_level in enumerate(noise_levels):
            print(f'noise level = {noise_level:.5f}')
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
            y_true,y_pred,y_prob = [],[],[]
            
            for batch_features,batch_labels in dataloader_test:
                (reconstruction,
                 extracted_features,
                 z,mu,log_var,
                 hidden_representation,
                 image_category) = vae(batch_features.to(device))
                y_true.append(batch_labels)
                y_pred.append(image_category.max(1)[1])
                y_prob.append(image_category.max(1)[0])
                hidden_representations.append(hidden_representation)
                sampled_representations.append(z.view(z.shape[0],-1))
                distances.append(np.diag(distance.cdist(hidden_representations[-1].detach().cpu().numpy(),
                                                        sampled_representations[-1].detach().cpu().numpy(),
                                                        metric='correlation')))
            y_true = torch.cat(y_true).detach().cpu().numpy()
            y_pred = torch.cat(y_pred).detach().cpu().numpy()
            y_prob = torch.cat(y_prob).detach().cpu().numpy()
            
            accuracy = np.sum(y_true == y_pred) / y_true.shape[0]
            
            results['noise_level'].append(noise_level)
            results['accuracy'].append(accuracy)
            
            for _y_true,_y_pred,_y_prob in zip(y_true,y_pred,y_prob):
                df_res['noise_level'].append(noise_level)
                df_res['y_true'].append(_y_true)
                df_res['y_pred'].append(_y_pred)
                df_res['y_prob'].append(_y_prob)
    hidden_representations = torch.cat(hidden_representations).detach().cpu().numpy()
    sampled_representations = torch.cat(sampled_representations).detach().cpu().numpy()
    distances = np.concatenate(distances)
    results = pd.DataFrame(results)
    df_res = pd.DataFrame(df_res)
    df_res['acc'] = df_res['y_true'] == df_res['y_pred']
    
    fig,axes = plt.subplots(figsize = (16,16),
                            nrows = 2,
                            ncols = 2,
                            sharey = True,
                            sharex = True,
                            )
    for (noise_level,matched),df_sub in df_res.groupby(['noise_level','acc']):
        if matched:
            axes[0][0] = sns.kdeplot(df_sub['y_prob'],alpha = .5,ax = axes[0][0])
            axes[0][1] = sns.kdeplot(distances[df_sub.index],alpha = .5,ax = axes[0][1])
        else:
            axes[1][0] = sns.kdeplot(df_sub['y_prob'],alpha = .5,ax = axes[1][0])
            axes[1][1] = sns.kdeplot(distances[df_sub.index],alpha = .5,ax = axes[1][1])