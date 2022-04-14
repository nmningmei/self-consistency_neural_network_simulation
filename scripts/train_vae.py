#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 13:00:57 2022

@author: nmei
"""
import torch
from torch import nn,optim
import numpy as np

from utils_deep import (hidden_activation_functions,
                        dataloader,
                        simple_augmentations
                        )
from vae_models import VanillaVAE



if __name__ == "__main__":
    dataset_name = 'CIFAR100'
    # image setting
    batch_size = 32
    image_resize = 32
    noise_level_train = 0.
    noise_level_test = 0.
    rotation = True
    gitter_color = False
    # set up random seeds and GPU/CPU
    torch.manual_seed(12345)
    np.random.seed(12345)
    if torch.cuda.is_available():torch.cuda.empty_cache()
    torch.cuda.manual_seed(12345)
    torch.cuda.manual_seed_all(12345)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model settings
    pretrained_model_name = 'vgg19'
    hidden_units = 300
    hidden_func_name = 'relu'
    hidden_activation = hidden_activation_functions(hidden_func_name)
    hidden_dropout = 0.
    latent_units = 256
    latent_func_name = 'leaky_relu'
    latent_activation = hidden_activation_functions(latent_func_name)
    latent_dropout = 0.
    hidden_dims = [256,128,64,32]
    model_args = dict(pretrained_model_name = pretrained_model_name,
                      hidden_units = hidden_units,
                      hidden_activation = hidden_activation,
                      hidden_dropout = hidden_dropout,
                      latent_units = latent_units,
                      latent_activation = latent_activation,
                      latent_dropout = latent_dropout,
                      hidden_dims = [256,128,64,32,16],
                      in_channels = 3,
                      in_shape = [1,3,image_resize,image_resize],
                      device = device,
                      )
    # trainer settings
    learning_rate = 1e-4
    l2_regularization = 1e-16
    print_train = True
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
                                                   batch_size = batch_size,
                                                   shuffle = True,
                                                   )
    dataloader_test,_ = dataloader(dataset_name = dataset_name,
                                   train = False,
                                   transform = transform,
                                   batch_size = batch_size,
                                   shuffle = True,
                                   )
    
    # build the variational autoencoder
    vae = VanillaVAE(**model_args).to(device)
    params = [p for p in vae.parameters() if p.requires_grad == True]
    recon_loss_func = nn.MSELoss()
    optimizer = optim.Adam(params,
                           lr = learning_rate,
                           weight_decay = l2_regularization,
                           )
    # train loop
    from tqdm import tqdm
    vae.train(True)
    idx_epoch = 0
    train_loss = 0.
    iterator = tqdm(enumerate(dataloader_train))
    for ii,(batch_features,batch_labels) in iterator:
        optimizer.zero_grad()
        reconstruction,hidden_representation,z,mu,log_var = vae(batch_features)
        recon_loss = recon_loss_func(batch_features,reconstruction)
        kld_loss = vae.kl_divergence(z, mu, log_var)
        loss_batch = recon_loss + kld_loss
        # backpropagation
        loss_batch.backward()
        # modify the weights
        optimizer.step()
        # record the loss of a mini-batch
        train_loss += loss_batch.data
        if print_train:
            iterator.set_description(f'epoch {idx_epoch+1}-{ii + 1:3.0f}/{100*(ii+1)/len(dataloader_train):2.3f}%,loss = {train_loss/(ii+1):.6f}')
        


