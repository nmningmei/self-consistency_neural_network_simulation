#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 13:00:57 2022

@author: nmei
"""
import os,torch
from torch import nn
from typing import Tuple
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.spatial import distance
from sklearn.metrics import roc_auc_score

from utils_deep import (dataloader,
                        simple_augmentations
                        )
from models import (vae_classifier)
import experiment_settings

from matplotlib import pyplot as plt

sns.set_style('whitegrid')

def set_ax(ax,
           xlabel:str = '',
           ylabel:str = '',
           title:str = '',
           xlim:Tuple = (None,None),
           ylim:Tuple = (None,None),
           **xargs
           ) -> None:
    """
    

    Parameters
    ----------
    ax : TYPE
        DESCRIPTION.
    xlabel : str, optional
        DESCRIPTION. The default is ''.
    ylabel : str, optional
        DESCRIPTION. The default is ''.
    title : str, optional
        DESCRIPTION. The default is ''.
    xlim : Tuple, optional
        DESCRIPTION. The default is (None,None).
    ylim : Tuple, optional
        DESCRIPTION. The default is (None,None).

    Returns
    -------
    None
        DESCRIPTION.

    """
    ax.set(xlabel = xlabel,ylabel = ylabel,title = title,
           xlim = xlim,ylim = ylim,
           **xargs)
    return None

def add_columns(df:pd.DataFrame) -> pd.DataFrame:
    """
    Add the experiment settings to the dataframe

    Parameters
    ----------
    df : pd.DataFrame
        DESCRIPTION.

    Returns
    -------
    df : pd.DataFrame
        DESCRIPTION.

    """
    df['CNN'] = experiment_settings.pretrained_model_name
    df['hidden_units'] = experiment_settings.hidden_units
    df['hidden_func_name'] = experiment_settings.hidden_func_name
    df['latent_func_name'] = experiment_settings.latent_func_names[0]
    df['hidden_dropout'] = experiment_settings.hidden_dropout
    df['retrain_encoder'] = experiment_settings.retrain_encoder
    df['deep_dense_layers'] = experiment_settings.multi_hidden_layer
    return df

def bin_by_roc_auc(x:float) -> int:
    """
    Bin the trials of the testing results by the ROC AUC scores

    Parameters
    ----------
    x : float
        DESCRIPTION.

    Returns
    -------
    int
        99 means a group of ROC AUC we are not interested at the moment.

    """
    if 0.65 > x >= 0.55:
        return 0
    elif 0.75 > x >= 0.65:
        return 1
    elif 0.85 > x >= 0.75:
        return 2
    elif 0.95 > x >= 0.85:
        return 3
    else:
        return 99

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
                               in_shape                 = [1,# dummy value
                                                           3,# n_channels
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
    kde_args                = dict(cut = 0,
                                   )
    
    # build the variational autoencoder
    print(f'Build CLF-VAE model on CNN backbone {experiment_settings.pretrained_model_name}')
    vae             = vae_classifier(**model_args).to(device)
    vae.load_state_dict(torch.load(experiment_settings.f_name,map_location = device))
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
        for ii,noise_level in enumerate(experiment_settings.noise_levels):
            print(f'noise level = {noise_level:.5f}')
            # make transforms
            transform                           = simple_augmentations(
                                                        image_resize    = experiment_settings.image_resize,
                                                        noise_level     = noise_level,
                                                        rotation        = experiment_settings.rotation,
                                                        gitter_color    = experiment_settings.gitter_color,
                                                        )
            dataloader_test,_                   = dataloader(
                                                dataset_name    = experiment_settings.dataset_name,
                                                root            = experiment_settings.test_root,
                                                train           = False,
                                                transform       = transform,
                                                batch_size      = experiment_settings.batch_size,
                                                shuffle         = True,
                                                )
            y_true,y_pred,y_prob = [],[],[]
            for _ in range(experiment_settings.n_experiment_runs):
                for batch_features,batch_labels in dataloader_test:
                    (reconstruction,
                     extracted_features,
                     z,mu,log_var,
                     hidden_representation,
                     image_category) = vae(batch_features.to(device))
                    batch_labels = torch.nn.functional.one_hot(batch_labels,num_classes = 2,)
                    y_true.append(batch_labels[:,-1])
                    y_pred.append(image_category.max(1)[1])
                    y_prob.append(image_category[:,-1])
                    hidden_representations.append(hidden_representation)
                    sampled_representations.append(reconstruction)
                    # a = hidden_representation.detach().cpu().numpy()
                    # a = a - a.mean(1).reshape(-1,1)
                    # b = reconstruction.detach().cpu().numpy()
                    # b = b - b.mean(1).reshape(-1,1)
                    # distances.append(np.diag(distance.cdist(a,b,metric = 'cosine',)))
                    distances.append(1-nn.CosineSimilarity(dim=1,)(hidden_representation,
                                                                  reconstruction))
            y_true = torch.cat(y_true).detach().cpu().numpy()
            y_pred = torch.cat(y_pred).detach().cpu().numpy()
            y_prob = torch.cat(y_prob).detach().cpu().numpy()
            
            # accuracy = np.sum(y_true == y_pred) / y_true.shape[0]
            accuracy = roc_auc_score(y_true,y_prob,)
            
            results['noise_level'].append(noise_level)
            results['accuracy'].append(accuracy)
            
            for _y_true,_y_pred,_y_prob in zip(y_true,y_pred,y_prob):
                df_res['noise_level'].append(noise_level)
                df_res['y_true'].append(_y_true)
                df_res['y_pred'].append(_y_pred)
                df_res['y_prob'].append(_y_prob)
    hidden_representations = torch.cat(hidden_representations).detach().cpu().numpy()
    sampled_representations = torch.cat(sampled_representations).detach().cpu().numpy()
    # distances = np.concatenate(distances)
    distances = torch.cat(distances).detach().cpu().numpy()
    # distances = np.log(distances)
    # dict -> dataframe
    results = pd.DataFrame(results)
    df_res = pd.DataFrame(df_res)
    results = add_columns(results)
    df_res = add_columns(df_res)
    # save
    results.to_csv(os.path.join(experiment_settings.results_dir,'performance.csv'),index = False)
    df_res.to_csv(os.path.join(experiment_settings.results_dir,'outputs.csv'),index = False)
    
    df_res['acc'] = df_res['y_true'] == df_res['y_pred']
    
    results['bins'] = results['accuracy'].apply(bin_by_roc_auc)
    trial_bins = {noise_level:group for noise_level,group in results[['noise_level','bins']].values}
    df_res['bins'] = df_res['noise_level'].map(trial_bins)
    
    
    for group,df_sub in df_res.groupby(['bins']):
        if group != 99:
            idx_correct = df_sub[df_sub['acc'] == True].index
            idx_incorrect = df_sub[df_sub['acc'] == False].index
            distance_correct = distances[idx_correct]
            distance_incorrect = distances[idx_incorrect]
            
            experimental_difference = distance_incorrect.mean() - distance_correct.mean()
            print(experimental_difference)
