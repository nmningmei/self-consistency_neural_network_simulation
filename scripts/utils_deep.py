#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 02:32:01 2021

@author: nmei
"""

import os
from glob        import glob
from tqdm        import tqdm
from collections import OrderedDict
import pandas as pd
import numpy  as np

import torch
from torch          import nn,no_grad
from torch.utils    import data
from torch.nn       import functional as F
from torch          import optim
from torch.autograd import Variable

from torchvision.datasets import ImageFolder
from torchvision          import transforms
from torchvision          import models as Tmodels

from sklearn                 import metrics
from sklearn.preprocessing   import StandardScaler
from sklearn.svm             import LinearSVC,SVC
from sklearn.decomposition   import PCA
from sklearn.pipeline        import make_pipeline
from sklearn.model_selection import StratifiedShuffleSplit,cross_validate,permutation_test_score
from sklearn.calibration     import CalibratedClassifierCV
from sklearn.linear_model    import LogisticRegression
from sklearn.utils           import shuffle as sk_shuffle
from sklearn.ensemble        import RandomForestClassifier
from sklearn.metrics         import roc_auc_score

try:
    from utils_common import (simple_augmentations,
                              data_loader,
                              simple_FCNN_validation_loop,
                              compute_image_loss,
                              compute_meta_loss,
                              create_meta_labels,
                              simple_FCNN
                              )
except:
    from shutil import copyfile
    copyfile('../common_functions.py','utils_common.py')
    from utils_common import (simple_augmentations,
                              data_loader,
                              simple_FCNN_validation_loop,
                              compute_image_loss,
                              compute_meta_loss,
                              create_meta_labels,
                              simple_FCNN
                              )

def resample_behavioral_estimate(y_true,y_pred,n_sampling = int(1e3),shuffle = False):
    """
    Estimate the ROC AUC by resampling methods
    
    Inputs
    ---
    y_true: true labels, (n_samples,) or (n_samples, 2), binarized
    y_pred: predicted labels, (n_samples,) or (n_samples, 2), better by softmaxed
    n_sampling: int, default = 1000
    shuffle: bool, default = False. When it is False, we estimate the empirical 
            ROC AUC; when it is True, we estimate the chance level ROC AUC
    Output
    ---
    scores: list, (n_sampling,)
    """
    from joblib import Parallel,delayed
    def _temp_func(idx_picked,shuffle = shuffle):
        if shuffle:
            _y_pred = sk_shuffle(y_pred)
            score = metrics.roc_auc_score(y_true[idx_picked],_y_pred[idx_picked])
        else:
            score = metrics.roc_auc_score(y_true[idx_picked], y_pred[idx_picked])
        return score
    scores = Parallel(n_jobs = -1,verbose = 0)(delayed(_temp_func)(**{
        'idx_picked':np.random.choice(y_true.shape[0],y_true.shape[0],replace = True),
        'shuffle':shuffle}) for _ in range(n_sampling))
    
    return scores

def sample_for_metatraining(pretrained_model_name,
                            f_name_first,
                            noise_for_sample,
                            common_xargs,
                            metacognitive_network   = None,
                            f_name_second           = None,
                            image_resize            = 128,
                            device                  = 'cpu',
                            train_root              = '',
                            batch_size              = 8,
                            image_loss_func         = nn.BCELoss(),
                            output_activation       = 'softmax',
                            n_meta_samples          = int(1e3),
                            sample_noise            = False,
                            print_sampling          = False,
                            get_hidden_for_clf      = False,
                            get_hidden_for_meta     = False,
                            n_clf_experiments       = 20,
                            ):
    if get_hidden_for_clf:
        print('sample for decoding image categories')
        features_classification,labels_classification = [],[]
        image_classifier        = simple_FCNN(
                                            pretrained_model_name   = pretrained_model_name,
                                            in_shape                = (1,3,image_resize,image_resize),
                                            **common_xargs
                                            ).to(device)
        # load the best model weights
        image_classifier.load_state_dict(torch.load(f_name_first,map_location = device))
        # freeze the model weights
        for p in image_classifier.parameters():p.requires_grad = False
        image_classifier.eval()
        # the sampling dataloader
        dataloader = data_loader(train_root,
                                 augmentations = simple_augmentations(image_resize,
                                                                      noise_level     = noise_for_sample,
                                                                      rotation        = True,
                                                                      gitter_color    = False,
                                                                      ),
                                 batch_size    = batch_size,
                                 )
        for _ in range(n_clf_experiments):
            with torch.no_grad():
                _,y_pred,y_true,features = simple_FCNN_validation_loop(
                                                    net                 = image_classifier,
                                                    loss_func           = image_loss_func,
                                                    dataloader          = dataloader,
                                                    device              = device,
                                                    output_activation   = output_activation,
                                                    verbose             = 0,
                                                    )
            y_pred      = torch.cat(y_pred).detach().cpu().numpy()
            y_true      = torch.cat(y_true).detach().cpu().numpy()
            features    = torch.cat(features).detach().cpu().numpy()
            features_classification.append(features)
            labels_classification.append(y_true[:,-1])
        features_classification = np.concatenate(features_classification)
        labels_classification = np.concatenate(labels_classification)
        print('done sampling')
    
    (features_correct,
     labels_correct,
     features_incorrect,
     labels_incorrect) = [],[],[],[]
    if get_hidden_for_meta:
        features_meta,labels_meta = [],[]
    if print_sampling:
        iterator = tqdm(range(int(1e5)))
    else:
        iterator = range(int(1e5))
    for _ in iterator:
        image_classifier        = simple_FCNN(
                                            pretrained_model_name   = pretrained_model_name,
                                            in_shape                = (1,3,image_resize,image_resize),
                                            **common_xargs
                                            ).to(device)
        # load the best model weights
        image_classifier.load_state_dict(torch.load(f_name_first,map_location = device))
        # freeze the model weights
        for p in image_classifier.parameters():p.requires_grad = False
        image_classifier.eval()
        if get_hidden_for_meta:
            metacognitive_network.load_state_dict(torch.load(f_name_second,map_location = device))
            for p in metacognitive_network.parameters():p.requires_grad = False
            metacognitive_network.eval()
        
        # the sampling dataloader
        dataloader = data_loader(train_root,
                                 augmentations = simple_augmentations(image_resize,
                                                                      noise_level     = noise_for_sample,
                                                                      rotation        = True,
                                                                      gitter_color    = False,
                                                                      ),
                                 batch_size    = batch_size,
                                 )
        with torch.no_grad():
            _,y_pred,y_true,features = simple_FCNN_validation_loop(
                                                net                 = image_classifier,
                                                loss_func           = image_loss_func,
                                                dataloader          = dataloader,
                                                device              = device,
                                                output_activation   = output_activation,
                                                verbose             = 0,
                                                )
            if get_hidden_for_meta:
                meta_hidden,meta_output = metacognitive_network(torch.cat(features))
        y_pred      = torch.cat(y_pred).detach().cpu().numpy()
        y_true      = torch.cat(y_true).detach().cpu().numpy()
        features    = torch.cat(features).detach().cpu().numpy()
        if get_hidden_for_meta:
            features_meta.append(meta_hidden)
            labels_meta.append(y_true[:,-1])
        if output_activation == 'softmax':
            labels                      = np.argmax(y_true,1)
            binarized_image_category    = np.argmax(y_pred,1)
        elif output_activation == 'sigmoid':
            binarized_image_category    = np.array(y_pred >= 0.5,dtype = np.float32)
        else:
            raise NotImplementedError
        
        meta_labels     = np.array(binarized_image_category == labels,dtype = np.float32)
        idx_incorrect,  = np.where(meta_labels == 0.)
        if len(idx_incorrect) > n_meta_samples:
            idx_incorrect = np.random.choice(idx_incorrect,
                                             size       = n_meta_samples,
                                             replace    = False,
                                             )
        if noise_for_sample > 0.5:
            idx_correct,    = np.where(meta_labels == 1.)
            if len(idx_correct) > len(idx_incorrect):
                idx_correct = np.random.choice(idx_correct,
                                               size     = len(idx_incorrect),
                                               replace  = False,
                                               )
        else:
            idx_correct     = np.random.choice(np.where(meta_labels == 1.)[0],
                                               size     = idx_incorrect.shape[0],
                                               replace  = False,
                                               )
        
        features_incorrect.append(  features[   idx_incorrect   ])
        labels_incorrect.append(    meta_labels[idx_incorrect   ])
        features_correct.append(    features[   idx_correct     ])
        labels_correct.append(      meta_labels[idx_correct     ])
        
        if print_sampling:
            iterator.set_description(f'in noise level {noise_for_sample:.4f}, has collected {np.concatenate(labels_incorrect).shape[0]}')
        
        if np.concatenate(labels_incorrect).shape[0] >= n_meta_samples:
            break
    
    features_incorrect  = np.concatenate(features_incorrect)
    labels_incorrect    = np.concatenate(labels_incorrect)
    features_correct    = np.concatenate(features_correct)
    labels_correct      = np.concatenate(labels_correct)
    
    # make sure the samples are the same for different noise levels
    features_incorrect  = features_incorrect[:n_meta_samples]
    labels_incorrect    = labels_incorrect[:n_meta_samples]
    features_correct    = features_correct[:n_meta_samples]
    labels_correct      = labels_correct[:n_meta_samples]
    
    if get_hidden_for_meta:
        features_meta   = np.concatenate(features_meta)
        labels_meta     = np.concatenate(labels_meta)
    
    if sample_noise:
        # sample some noise examples
        dataloader = data_loader(data_root = train_root,
                                 augmentations = simple_augmentations(image_resize  = image_resize,
                                                                      noise_level   = noise_for_sample,
                                                                      rotation      = True,
                                                                      gitter_color  = False,),
                                 batch_size = batch_size,
                                 )
        with torch.no_grad():
            features_noise = []
            for _features,_labels in dataloader:
                noise_distribution      = torch.distributions.normal.Normal(
                                            _features.mean(),
                                            _features.std(),
                                            )
                noise_images            = noise_distribution.sample(_features.shape)
                _,hidden_representation = image_classifier(noise_images.to(device))
                features_noise.append(hidden_representation)
        features_noise  = torch.cat(features_noise).detach().cpu().numpy()
        _idx            = np.random.choice(features_noise.shape[0],size = int(n_meta_samples / 10),replace = False)
        features_noise  = features_noise[_idx]
    
    # concatenate the features and labels
    if sample_noise:
        sampled_features    = np.concatenate([features_correct,
                                              features_incorrect,
                                              features_noise
                                              ])
        sampled_labels      = np.concatenate([labels_correct,
                                              labels_incorrect,
                                              np.array([0.5] * features_noise.shape[0])
                                              ])
    else:
        sampled_features    = np.concatenate([features_correct,
                                              features_incorrect,
                                              ])
        sampled_labels      = np.concatenate([labels_correct,
                                              labels_incorrect,
                                              ])
    sampled_features,sampled_labels = sk_shuffle(sampled_features,sampled_labels)
    
    image_classifier.to('cpu')
    del image_classifier
    
    outputs = []
    outputs.append(sampled_features)
    outputs.append(sampled_labels)
    if get_hidden_for_clf:
        outputs.append(features_classification)
        outputs.append(labels_classification)
    if get_hidden_for_meta:
        outputs.append(features_meta)
        outputs.append(labels_meta)
    
    return outputs

def get_responses(valid_root = '',
                  image_resize = 128,
                  noise_level = None,
                  batch_size = 8,
                  n_experiment_runs = 20,
                  image_classifier = None,
                  metacognitive_network = None,
                  device = 'cpu',
                  ):
    """
    
    """
    dataloader        = data_loader(
            valid_root,
            augmentations   = simple_augmentations(image_resize,
                                                   noise_level  = noise_level,
                                                   rotation     = True,
                                                   gitter_color = False,
                                                   ),
            batch_size      = batch_size,
            )
    with torch.no_grad():
        y_true,y_pred,y_conf = [],[],[]
        for _ in range(n_experiment_runs):
            _y_true,_y_pred,_y_conf = [],[],[]
            for input_images,input_labels in dataloader:
                image_category,hidden_representation = image_classifier(input_images.to(device))
                _,meta_prediction = metacognitive_network(hidden_representation)
                _y_true.append(input_labels)
                _y_pred.append(image_category)
                _y_conf.append(meta_prediction)
            y_true.append(torch.cat(_y_true))
            y_pred.append(torch.cat(_y_pred))
            y_conf.append(torch.cat(_y_conf))
        y_true = torch.cat(y_true).detach().cpu().numpy()
        y_true = np.array([1 - y_true, y_true]).T
        y_pred = torch.cat(y_pred).detach().cpu().numpy()
        # acc    = np.array(y_true == y_pred.argmax(1),dtype = np.float32)
        y_conf = torch.cat(y_conf).detach().cpu().numpy()
        df = pd.DataFrame(np.hstack([y_true,y_pred,y_conf]),
                          columns = ['True label of Living',
                                     'True label of nonLiving',
                                     'Probability of Living',
                                     'Probability of nonLiving',
                                     'Probability of incorrect',
                                     'Probability of correct',
                                     ]
                          )
        df['noise level'] = noise_level
    return df

def metatrain_loop(metacognitive_network,
                   meta_loss_func,
                   optimizer,
                   device,
                   train_loader,
                   noise_level          = None,
                   output_activation    = 'softmax',
                   idx_epoch            = 1,
                   print_train          = False,
                   l1_term              = 1e-5,
                   ):
    """
    
    """
    metacognitive_network.to(device).train(True)
    train_loss = 0.
    # verbose level
    if print_train:
        iterator = tqdm(enumerate(train_loader))
    else:
        iterator = enumerate(train_loader)
    
    for ii,(features,labels) in iterator:
        # put the features and labels to memory
        features            = Variable(features).to(device)
        labels              = labels.to(device)
        
        # zero gradient
        optimizer.zero_grad()
        # forward pass of the metacognitive network
        meta_hidden_representation,meta_output = metacognitive_network(features)
        # loss
        temp_loss = compute_meta_loss(meta_loss_func,meta_output,labels,device)
        # sparse meta-prediction loss if softmax
        sparse_loss = l1_term * torch.norm(meta_output,1,dim = 0).min()
        temp_loss += sparse_loss
        # backward
        temp_loss.backward()
        # update
        optimizer.step()
        # 
        train_loss += temp_loss
        if print_train:
            iterator.set_description(f'epoch {idx_epoch+1}-{ii + 1:3.0f}/{100*(ii+1)/len(train_loader):2.3f}%,loss = {train_loss/(ii+1):.6f}')
            
    return train_loss/(ii+1)

def metavalidation_loop(metacognitive_network,
                        meta_loss_func,
                        device,
                        valid_loader,
                        noise_level         = None,
                        output_activation   = 'softmax', # true=softmax,false=sigmoid
                        verbose             = 0,
                        rotation            = True,
                        gitter_color        = False,
                        ):
    """
    
    """
    # specify the gradient being frozen and dropout etc layers will be turned off
    metacognitive_network.to(device).eval()
    with no_grad():
        valid_loss      = 0.
        y_conf          = []
        meta_label      = []
        if verbose == 0:
            iterator = enumerate(valid_loader)
        else:
            iterator         = tqdm(enumerate(valid_loader))
        for ii,(batch_features,batch_labels) in iterator:
            batch_labels.to(device)
            temp_loss = 0
            if ii + 1 <= len(valid_loader):
                # reshape the labels for the image predictions
                
                # put the features and labels to memory
                batch_features = Variable(batch_features).to(device)
                batch_labels  = batch_labels.to(device)
                # forward pass of the metacognitive network
                meta_hidden_representation,meta_output = metacognitive_network(batch_features)
                # loss
                temp_loss = compute_meta_loss(meta_loss_func,meta_output,batch_labels,device)
                #
                valid_loss += temp_loss
                denominator = ii
                
                y_conf.append(meta_output)
                meta_label.append(batch_labels)
        valid_loss = valid_loss / (denominator + 1)
    return valid_loss,y_conf,meta_label

def metatrain_and_metavalidation(
        metacognitive_network,
        f_name_second,
        meta_loss_func,
        optimizer,
        output_activation   = 'softmax',
        device              = 'cpu',
        n_epochs            = int(3e3),
        print_train         = True,
        patience            = 5,
        train_loader        = None,
        valid_loader        = None,
        tol                 = 1e-4,
        warmup_epochs       = 5,
        ):
    """
    
    """
    torch.random.manual_seed(12345)
    metacognitive_network.to(device)
    
    best_valid_loss     = np.inf
    losses              = []
    counts              = 0
    mat_score           = 0
    for idx_epoch in range(n_epochs):
        # train
        print('\ntraining ...')
        _               = metatrain_loop(
        metacognitive_network   = metacognitive_network,
        meta_loss_func          = meta_loss_func,
        optimizer               = optimizer,
        train_loader            = train_loader,
        device                  = device,
        idx_epoch               = idx_epoch,
        print_train             = print_train,
        output_activation       = output_activation,
        )
        print('\nvalidating ...')
        with torch.no_grad():
            valid_loss,y_conf,meta_label = metavalidation_loop(
            metacognitive_network   = metacognitive_network,
            meta_loss_func          = meta_loss_func,
            output_activation       = output_activation,
            valid_loader            = valid_loader,
            device                  = device,
            )
        y_conf = torch.cat(y_conf).detach().cpu().numpy()
        meta_labels = torch.cat(meta_label).detach().cpu().numpy()
        meta_labels = np.array([1 - meta_labels,meta_labels]).T
        idx_no_noise, = np.where(meta_labels[:,-1] != 0.5)
        mat_score = metrics.roc_auc_score(meta_labels[idx_no_noise],
                                          y_conf[idx_no_noise],
                                          )
        if idx_epoch > warmup_epochs: # warming up
            temp = valid_loss.cpu().clone().detach().type(torch.float64)
            if np.logical_and(temp < best_valid_loss,np.abs(best_valid_loss - temp) >= tol):
                best_valid_loss = valid_loss.cpu().clone().detach().type(torch.float64)
                torch.save(metacognitive_network.state_dict(),f_name_second)# why do i need state_dict()?
                counts = 0
            else:
                counts += 1
        losses.append(best_valid_loss)
        _idx = np.random.choice(len(y_conf),size = 1)
        print(f'epoch {idx_epoch + 1}, loss = {valid_loss:6f},meta = {mat_score:.4f},count = {counts}')
        print(y_conf[_idx])
        print(meta_labels[_idx])
        if counts >= patience:#(len(losses) > patience) and (len(set(losses[-patience:])) == 1):
            break
    return metacognitive_network,losses
