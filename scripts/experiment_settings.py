#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 18:12:04 2022

@author: nmei
"""
import os
import numpy as np
from utils_deep import hidden_activation_functions

dataset_name            = None
train_root              = '../data/greyscaled'
valid_root              = '../data/Konklab'
test_root               = '../data/metasema_images'

# image setting
batch_size              = 16 # batch size for each epoch
image_resize            = 128 # image hight
noise_level_train       = 1e-3 # noise level in training
noise_level_test        = 0. # noise level in testing
rotation                = True # image augmentation
gitter_color            = False # image augmentation for Gabor patches

# model settings
pretrained_model_name   = 'vgg19'
hidden_units            = 10 # hidden layer units
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
vae_out_func_name       = hidden_func_name # activation function of the reconstruction
vae_output_activation   = hidden_activation_functions(vae_out_func_name)
retrain_encoder         = True # retrain the CNN backbone convolutional layers
multi_hidden_layer      = False # add more dense layer to the classifier and the VAE encoder
# customize
experiment_name         = 'clf+vae' + '_{}_{}_{}_{}_{}'.format(
                                                pretrained_model_name,
                                                hidden_units,
                                                hidden_func_name,
                                                latent_func_names[0],
                                                hidden_dropout,)
model_dir               = os.path.join('../models',experiment_name)
figure_dir              = os.path.join('../figures',experiment_name)
results_dir             = os.path.join('../results',experiment_name)
f_name                  = os.path.join(model_dir,'clf-vae_{}_{}_{}_{}_{}.h5'.format(
                                                             pretrained_model_name,
                                                             hidden_units,
                                                             hidden_func_name,
                                                             latent_func_names[0],
                                                             hidden_dropout,
                                                             ))
if multi_hidden_layer:
    f_name              = f_name.replace('.h5','_deep.h5')
    experiment_name     = experiment_name + '+deep'
latent_units            = hidden_dims[-1] if multi_hidden_layer else int(hidden_units/2)
# train settings
learning_rate           = 1e-4 # initial learning rate, will be reduced by 10 after warmup epochs
l2_regularization       = 1e-4 # L2 regularization term, used as weight decay
print_train             = True # print the progresses
n_epochs                = int(1e3) # max number of epochs
warmup_epochs           = 10 # we don't save the models in these epochs
patience                = 20 # we wait for a number of epochs after the best performance
tol                     = 1e-4 # the difference between the current best and the next best
n_noise                 = int(batch_size/2) # number of noisy images used in training the classifier
retrain                 = True # retrain the VAE
# testing settings
n_noise_levels          = 50
min_noise_level         = np.log10(0.01)
max_noise_level         = np.log10(1000)
noise_levels            = np.concatenate([[0],np.logspace(min_noise_level,max_noise_level,n_noise_levels)])

if __name__ == "__main__":
    os.system("python3 train_vae-clf.py")
    os.system("python3 test_vae-clf.py")