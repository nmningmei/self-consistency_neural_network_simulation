#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 14:44:35 2021

@author: nmei
"""

import os,gc
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


##########################################
output_act_func_dict = {'softmax':nn.Softmax(dim = 1), # softmax dim = 1
                        'sigmoid':nn.Sigmoid(),
                        'hinge':torch.nn.HingeEmbeddingLoss}
probability_func_dict = {'softmax':F.softmax,    # softmax dim = 1
                         'sigmoid':torch.sigmoid}
softmax_dim = 1
##########################################
def garbor_generator(image_size = 128,lamda = 4, thetaRad_base = 45,figure_dir = '../data/gabors'):
    """
    Inputs
    -------------
    image_size: int, hight and width of the images
    lamda: float, better be in range between 4 and 32
    thetaRad_base:float, base value of theta, in degrees
    figure_dir: string, mother directory of the gabors, will be divided into train
            and validation
    """
    if (np.abs(np.abs(thetaRad_base) - 45) < 5): # make easy cases the validation set
        folder = 'validation'
    else:
        folder = 'train'
    # identify left and right based on the theta base values
    if thetaRad_base < 0:
        sub_folder = 'left'
    else:
        sub_folder = 'right'
    # generate the file name based on the parameteres
    name = f'{lamda}_{thetaRad_base}_{sub_folder}.jpg'
    saving_name = os.path.join(figure_dir,folder,sub_folder,name)
    if not os.path.exists(os.path.join(figure_dir,folder,sub_folder)):
        os.makedirs(os.path.join(figure_dir,folder,sub_folder))
    # convert degree to pi-based
    thetaRad = thetaRad_base * np.pi / 180
    # Sanjeev's algorithm
    X       = np.arange(image_size)
    X0      = (X / image_size) - .5
    freq    = image_size / lamda
    Xf      = X0 * freq * 2 * np.pi
    sinX    = np.sin(Xf)
    Xm,Ym   = np.meshgrid(X0,X0)
    Xt      = Xm * np.cos(thetaRad)
    Yt      = Ym * np.sin(thetaRad)
    XYt     = Xt + Yt
    XYf     = XYt * freq * 2 * np.pi

    grating = np.sin(XYf)

    s       = 0.075
    w       = np.exp(-(0.3 * ((Xm**2) + (Ym**2)) / (2 * s**2))) * 2
    w[w > 1]= 1
    gabor   = ((grating - 0.5) * w) + 0.5
    from matplotlib import pyplot as plt
    plt.close('all')
    fig,ax = plt.subplots(figsize = (6,6))
    ax.imshow(gabor,cmap = plt.cm.gray)
    ax.axis('off')
    plt.close('all')
    fig.savefig(saving_name,dpi = 100,bbox_inches = 'tight')
    gc.collect() # delete memory garbages

def bin_array(x):
    """
    categorize the confidence outputs
    """
    if 0.25 >= x >= 0:
        return 1
    elif 0.5 >= x > 0.25:
        return 2
    elif 0.75 >= x > 0.5:
        return 3
    elif 1 >= x > 0.75:
        return 4
    else:
        raise ValueError

#candidate models
def candidates(model_name,pretrained = True,):
    picked_models = dict(
            resnet18        = Tmodels.resnet18(pretrained           = pretrained,
                                              progress              = False,),
            alexnet         = Tmodels.alexnet(pretrained            = pretrained,
                                             progress               = False,),
            # squeezenet      = Tmodels.squeezenet1_1(pretrained      = pretrained,
            #                                        progress         = False,),
            vgg19           = Tmodels.vgg19_bn(pretrained           = pretrained,
                                              progress              = False,),
            densenet169     = Tmodels.densenet169(pretrained        = pretrained,
                                                 progress           = False,),
            inception       = Tmodels.inception_v3(pretrained       = pretrained,
                                                  progress          = False,),
            # googlenet       = Tmodels.googlenet(pretrained          = pretrained,
            #                                    progress             = False,),
            # shufflenet      = Tmodels.shufflenet_v2_x0_5(pretrained = pretrained,
            #                                             progress    = False,),
            mobilenet       = Tmodels.mobilenet_v2(pretrained       = pretrained,
                                                  progress          = False,),
            # resnext50_32x4d = Tmodels.resnext50_32x4d(pretrained    = pretrained,
            #                                          progress       = False,),
            resnet50        = Tmodels.resnet50(pretrained           = pretrained,
                                              progress              = False,),
            )
    return picked_models[model_name]

def noise_fuc(x,noise_level = 1):
    """
    add guassian noise to the images during agumentation procedures

    Inputs
    --------------------
    x: torch.tensor, batch_size x 3 x height x width
    noise_level: float, standard deviation of the gaussian distribution
    """
    generator = torch.distributions.normal.Normal(0,noise_level)
    return x + generator.sample(x.shape)

def Find_Optimal_Cutoff(target, predicted):
    """ Find the optimal probability cutoff point for a classification model related to event rate
    Parameters
    ----------
    target : Matrix with dependent or target data, where rows are observations

    predicted : Matrix with predicted data, where rows are observations

    Returns
    -------     
    list type, with optimal cutoff value

    """
    from sklearn.metrics import roc_curve
    fpr, tpr, threshold         = roc_curve(target, predicted)
    i                           = np.arange(len(tpr)) 
    roc                         = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'threshold' : pd.Series(threshold, index=i)})
    roc_t                       = roc.iloc[(roc.tf-0).abs().argsort()[:1]]

    return list(roc_t['threshold'])[0]

def simple_augmentations(image_resize   = 128,
                         noise_level    = None,
                         rotation       = True,
                         gitter_color   = False,
                         ):
    """
    Simple augmentation steps
    
    Inputs 
    ---
    image_resize: int, the height and width of the images
    noise_level: float, standard deviation of the Gaussian distribution the noise is sampled from
    rotation: bool, one of the augmentation methods, for object recognition only
    gitter_color: bool, one of the augmentation methods, for Gabor only
    
    Outputs
    ---
    torchvision.transformer object
    """
    steps = [transforms.Resize((image_resize,image_resize)),]
    
    if rotation and not gitter_color:
        steps.append(transforms.RandomHorizontalFlip(p = 0.5))
        steps.append(transforms.RandomRotation(45,))
        steps.append(transforms.RandomVerticalFlip(p = 0.5))
    elif gitter_color and not rotation:
        steps.append(transforms.RandomCrop((image_resize,image_resize)))
        steps.append(transforms.ColorJitter(brightness = 0.25,
                                            contrast = 0.25,
                                            saturation = 0.25,
                                            hue = 0.25,))
    
    steps.append(transforms.ToTensor())
    if noise_level > 0:
        steps.append(transforms.Lambda(lambda x:noise_fuc(x,noise_level)))
    steps.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    # mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    transform_steps = transforms.Compose(steps)
    return transform_steps

class customizedDataset(ImageFolder):
    def __getitem__(self, idx):
        original_tuple  = super(customizedDataset,self).__getitem__(idx)
        path = self.imgs[idx][0]
        tuple_with_path = (original_tuple +  (path,))
        return tuple_with_path

def freeze_layer_weights(layer):
    for param in layer.parameters():
        param.requries_grad = False
    return None

def data_loader(data_root:str,
                augmentations:transforms    = None,
                batch_size:int              = 8,
                num_workers:int             = 2,
                shuffle:bool                = True,
                return_path:bool            = False,
                )->data.DataLoader:
    """
    Create a batch data loader from a given image folder.
    The folder must be organized as follows:
        main ---
             |
             -----class 1 ---
                         |
                         ----- image 1.jpeg
                         .
                         .
                         .
            |
            -----class 2 ---
                        |
                        ---- image 1.jpeg
                        .
                        .
                        .
            |
            -----class 3 ---
                        |
                        ---- image 1.jpeg
                        .
                        .
                        .
    Input
    --------------------------
    data_root: str, the main folder
    augmentations: torchvision.transformers.Compose, steps of augmentation
    batch_size: int, batch size
    num_workers: int, CPU --> GPU carrier, number of CPUs
    shuffle: Boolean, whether to shuffle the order
    return_pth: Boolean, lod the image paths

    Output
    --------------------------
    loader: DataLoader, a Pytorch dataloader object
    """
    if return_path:
        datasets = customizedDataset(
                root                        = data_root,
                transform                   = augmentations
                )
    else:
        datasets    = ImageFolder(
                root                        = data_root,
                transform                   = augmentations
                )
    loader      = data.DataLoader(
                datasets,
                batch_size                  = batch_size,
                num_workers                 = num_workers,
                shuffle                     = shuffle,
                )
    return loader

def createLossAndOptimizer(net,
                           learning_rate:float = 1e-4, 
                           params = None,
                           l2_penalty_term:float = 1e-6):
    """
    To create the loss function and the optimizer

    Inputs
    ----------------
    net: nn.Module, torch model class containing parameters method
    learning_rate: float, learning rate
    params: None or list of nn.Module.parameters, generator object
    l2_penalty_term: float, must be > 0, L2 regrularization by ADAM

    Outputs
    ----------------
    loss: nn.Module, loss function
    optimizer: torch.optim, optimizer
    """
    #Loss function
    loss        = nn.BCELoss()
    #Optimizer
    if params == None:
        optimizer   = optim.Adam([params for params in net.parameters()],
                                  lr = learning_rate,
                                  weight_decay = l2_penalty_term,)
    else:
        optimizer   = optim.Adam(params,
                                 lr = learning_rate,
                                 weight_decay = l2_penalty_term,)

    return(loss, optimizer)

def define_type(model_name):
    model_type          = dict(
            alexnet     = 'simple',
            vgg19       = 'simple',
            densenet169 = 'simple',
            inception   = 'inception',
            mobilenet   = 'simple',
            resnet18    = 'resnet',
            resnet50    = 'resnet',
            )
    return model_type[model_name]

def hidden_activation_functions(activation_func_name):
    funcs = dict(relu = nn.ReLU(),
                 selu = nn.SELU(),
                 elu = nn.ELU(),
                 sigmoid = nn.Sigmoid(),
                 tanh = nn.Tanh(),
                 linear = None,
                 )
    return funcs[activation_func_name]

def make_decoder(decoder_name,n_jobs = 1,):
    """
    Make decoders for the hidden representations

    Inputs
    ---------------
    decoder_name: String, to call the dictionary
    n_jobs: int, parallel argument
    """
    np.random.seed(12345)
    # linear SVM
    lsvm = LinearSVC(penalty        = 'l2', # default
                     dual           = True, # default
                     tol            = 1e-4, # not default
                     random_state   = 12345, # not default
                     max_iter       = int(1e4), # no default
                     class_weight   = 'balanced', # not default
                     )
    # to make the probabilistic predictions from the SVM
    lsvm = CalibratedClassifierCV(
                     base_estimator = lsvm,
                     method         = 'sigmoid',
                     cv             = 8,
                     )

    # RBF SVM
    svm = SVC(
                     tol            = 1e-3,
                     random_state   = 12345,
                     max_iter       = int(1e3),
                     class_weight   = 'balanced',
                     )
    # to make the probabilistic predictions from the SVM
    svm = CalibratedClassifierCV(
                     base_estimator = svm,
                     method         = 'sigmoid',
                     cv             = 8,
                     )
    
    # random forest implemented by scikit-learn
    rfc = RandomForestClassifier(
                      n_estimators  = int(1e3), # not default
                      bootstrap     = True,# not default
                      max_samples   = .9,# not default
                      random_state  = 12345,# not default
                      n_jobs        = n_jobs,# not default
                      verbose       = 0,
                      )

    # logistic regression
    logitstic = LogisticRegression(random_state = 12345)

    if decoder_name == 'linear-SVM':
        decoder = make_pipeline(StandardScaler(),
                                lsvm,
                                )
    elif decoder_name == 'PCA-linear-SVM':
        decoder = make_pipeline(StandardScaler(),
                                PCA(random_state = 12345,),
                                lsvm,)
    elif decoder_name == 'RBF-SVM':
        decoder = make_pipeline(StandardScaler(),
                                svm,
                                )
    elif decoder_name == 'RF':
        decoder = make_pipeline(StandardScaler(),
                                rfc,
                                )
    elif decoder_name == 'logit':
        decoder = make_pipeline(StandardScaler(),
                                logitstic,
                                )
    return decoder

def decode_hidden_layer(decoder,
                        features,
                        labels,
                        cv                  = None,
                        groups              = None,
                        n_splits            = 50,
                        test_size           = .2,):
    """
    Decode the hidden layer outputs from a convolutional neural network by a scikit-learn classifier

    Inputs
    -----------------------
    decoder: scikit-learn object
    features: numpy narray, n_sample x n_features
    labels: numpy narray, n_samples x 1
    cv: scikit-learn object or None, default being sklearn.model_selection.StratifiedShuffleSplit
    n_splits: int, number of cross validation
    test_size: float, between 0 and 1.
    """
    if cv == None:
        cv = StratifiedShuffleSplit(n_splits        = n_splits,
                                    test_size       = test_size,
                                    random_state    = 12345,
                                    )
        print(f'CV not defined, use StratifiedShuffleSplit(n_splits = {n_splits})')

    res = cross_validate(decoder,
                         features,
                         labels,
                         groups             = groups,
                         cv                 = cv,
                         scoring            = 'roc_auc',
                         n_jobs             = -1,
                         verbose            = 1,
                         return_estimator   = True,
                         )
    # plase uncomment below and test this when you have enough computational power, i.e. parallel in more than 16 CPUs
    pval = resample_ttest(res['test_score'],baseline = 0.5,n_permutation = int(1e4),
                                  one_tail = True,n_jobs = -1.)
    return res,cv,pval

def resample_ttest(x,
                   baseline         = 0.5,
                   n_permutation    = 10000,
                   one_tail         = False,
                   n_jobs           = 12,
                   verbose          = 0,
                   full_size        = True,
                   metric_func      = np.mean,
                   ):
    """
    http://www.stat.ucla.edu/~rgould/110as02/bshypothesis.pdf
    https://www.tau.ac.il/~saharon/StatisticsSeminar_files/Hypothesis.pdf
    Inputs:
    ----------
    x: numpy array vector, the data that is to be compared
    baseline: the single point that we compare the data with
    n_ps: number of p values we want to estimate
    one_tail: whether to perform one-tailed comparison
    """
    import numpy as np
    # import gc
    # from joblib import Parallel,delayed
    # statistics with the original data distribution
    t_experiment    = metric_func(x)
    null            = x - metric_func(x) + baseline # shift the mean to the baseline but keep the distribution

    if null.shape[0] > int(1e4): # catch for big data
        full_size   = False
    if not full_size:
        size        = (int(1e3),n_permutation)
    else:
        size = (null.shape[0],n_permutation)
    
    null_dist = np.random.choice(null,size = size,replace = True)
    t_null = metric_func(null_dist,0)
    
    if one_tail:
        return ((np.sum(t_null >= t_experiment)) + 1) / (size[1] + 1)
    else:
        return ((np.sum(np.abs(t_null) >= np.abs(t_experiment))) + 1) / (size[1] + 1) /2

    
def resample_ttest_2sample(a,b,
                           n_permutation        = 10000,
                           one_tail             = False,
                           match_sample_size    = True,
                           n_jobs               = 6,
                           verbose              = 0):
    from joblib import Parallel,delayed
    import gc
    # when the samples are dependent just simply test the pairwise difference against 0
    # which is a one sample comparison problem
    if match_sample_size:
        difference  = a - b
        ps          = resample_ttest(difference,
                                     baseline       = 0,
                                     n_permutation  = n_permutation,
                                     one_tail       = one_tail,
                                     n_jobs         = n_jobs,
                                     verbose        = verbose,)
        return ps
    else: # when the samples are independent
        t_experiment        = np.mean(a) - np.mean(b)
        if not one_tail:
            t_experiment    = np.abs(t_experiment)
        def t_statistics(a,b):
            group           = np.concatenate([a,b])
            np.random.shuffle(group)
            new_a           = group[:a.shape[0]]
            new_b           = group[a.shape[0]:]
            t_null          = np.mean(new_a) - np.mean(new_b)
            if not one_tail:
                t_null      = np.abs(t_null)
            return t_null
        try:
           gc.collect()
           t_null_null = Parallel(n_jobs = n_jobs,verbose = verbose)(delayed(t_statistics)(**{
                            'a':a,
                            'b':b}) for i in range(n_permutation))
        except:
            t_null_null = np.zeros(n_permutation)
            for ii in range(n_permutation):
                t_null_null = t_statistics(a,b)
        if one_tail:
            ps = ((np.sum(t_null_null >= t_experiment)) + 1) / (n_permutation + 1)
        else:
            ps = ((np.sum(np.abs(t_null_null) >= np.abs(t_experiment))) + 1) / (n_permutation + 1) / 2
        return ps

def make_meta_classes():
    temp = np.array([[0,0,0], # incorrect & living
                     [0,1,1], # incorrect & nonliving
                     [1,0,2], # correct   & living
                     [1,1,3], # correct   & nonliving
                     ])
    df = pd.DataFrame(temp,columns = ['corr','y','class'])
    return df

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

def internal_connection(
                        layer_type,
                        input_units,
                        output_units,
                        output_dropout,
                        output_activation,
                        device,
                        ):
    """
    create a linear hidden layer
    
    Inputs
    ---
    layer_type: str, default = "linear"
    input_units: int, in_features of the layer
    output_units: int, out_features of the layer
    output_drop: float, between 0 and 1
    output_activation: nn.Module or None, torch activation functions, None = linear activation function
    device: str or torch.device
    
    Outputs
    ---
    hidden_layer: nn.Sequential module
    """
    if layer_type == 'linear':
        latent_layer     = nn.Linear(input_units,output_units).to(device)
        dropout          = nn.Dropout(p = output_dropout).to(device)
        
        if output_activation is not None:
            hidden_layer = nn.Sequential(
                                latent_layer,
                                output_activation,
                                dropout)
        else:
            hidden_layer = nn.Sequential(
                                latent_layer,
                                dropout)
        return hidden_layer
    else:
        raise NotImplementedError

def simple_FCNN_train_loop(
               net,
               loss_func,
               optimizer,
               dataloader,
               device,
               output_activation    = 'softmax',
               idx_epoch            = 1,
               print_train          = False,
               l2_lambda            = 0,
               l1_lambda            = 0,
               n_noise              = 0,
               ):
    """
    A for-loop of train the autoencoder for 1 epoch

    Inputs
    -----------
    net: nn.Module, torch model class containing parameters method
    loss_func: nn.Module, loss function
    optimizer: torch.optim, optimizer
    dataloader: torch.data.DataLoader
    device:str or torch.device, where the training happens
    output_activation:str, the output activation function
    idx_epoch:int, for print
    print_train:Boolean, debug tool
    l2_lambda:float, L2 regularization lambda term
    l1_lambda:float, L1 regularization lambdd term
    n_noise: int, number of noise images to add to the training batch

    Outputs
    ----------------
    train_loss: torch.Float, average training loss
    net: nn.Module, the updated model

    """
    
    train_loss = 0.
    # set the model to "train"
    net.to(device).train(True)
    # verbose level
    iterator   = tqdm(enumerate(dataloader)) if print_train else enumerate(dataloader)
    
    for ii,(features,labels) in iterator:
        
        if n_noise > 0:
            # in order to have desired classification behavior, which is to predict
            # chance when no signal is present, we manually add some noise samples
            noise_generator = torch.distributions.normal.Normal(features.mean(),
                                                                features.std())
            noisy_features  = noise_generator.sample(features.shape)[:n_noise]
            noisy_labels    = torch.tensor([0.5] * labels.shape[0])[:n_noise]
            
            features        = torch.cat([features,noisy_features])
            labels          = torch.cat([labels,noisy_labels])
        
        # shuffle the training batch
        np.random.seed(12345)
        idx_shuffle         = np.random.choice(features.shape[0],features.shape[0],replace = False)
        features            = features[idx_shuffle]
        labels              = labels[idx_shuffle]
        
        if ii + 1 <= len(dataloader): # drop last
            # load the data to memory
            inputs      = Variable(features).to(device)
            # one of the most important steps, reset the gradients
            optimizer.zero_grad()
            # compute the outputs
            outputs,_   = net(inputs)
            # compute the losses
            if output_activation == 'softmax':
                labels  = torch.stack([1- labels,labels]).T
            
            loss_batch = compute_image_loss(loss_func,
                                            outputs.to(device),
                                            labels.view(outputs.shape).to(device),
                                            device)
            # sparse prediction loss if softmax
            if output_activation == 'softmax':
                sparse_loss = 1e-4 * torch.norm(outputs,1,dim = 0).min()
            else:
                sparse_loss = 1e-4 * torch.norm(outputs,1)
            loss_batch += sparse_loss
            # backpropagation
            loss_batch.backward()
            # modify the weights
            optimizer.step()
            # record the training loss of a mini-batch
            train_loss  += loss_batch.data
            if print_train:
                iterator.set_description(f'epoch {idx_epoch+1}-{ii + 1:3.0f}/{100*(ii+1)/len(dataloader):2.3f}%,loss = {train_loss/(ii+1):.6f}')
                
    return train_loss/(ii+1)

def simple_FCNN_validation_loop(
                    net,
                    loss_func,
                    dataloader,
                    device,
                    output_activation   = 'softmx',
                    verbose             = 0,
                    ):
    """
    Inputs
    ---
    net:nn.Module, torch model object
    loss_func:nn.Module, loss function
    dataloader:torch.data.DataLoader
    device:str or torch.device
    output_activation:str,
    
    Outputs
    ---
    valid_loss: float, validation loss value
    y_true: ndarray, (n_samples,) or (n_samples,2)
    y_pred: ndarray, (n_samples,) or (n_samples,2)
    features: ndarray, (n_samples,n_features), hidden representation features
    """
    
    # specify the gradient being frozen and dropout etc layers will be turned off
    net.to(device).eval()
    with no_grad():
        valid_loss      = 0.
        y_pred          = []
        y_true          = []
        features        = []
        iterator        = enumerate(dataloader) if verbose == 0 else tqdm(enumerate(dataloader))
        
        for ii,(batch_features,batch_labels) in iterator:
            batch_labels.to(device)
            if ii + 1 <= len(dataloader):
                # load the data to memory
                inputs             = Variable(batch_features).to(device)
                # compute the outputs
                outputs,_feature   = net(inputs)
                
                if output_activation == 'softmax':
                    batch_labels   = torch.stack([1 - batch_labels,batch_labels]).T
                
                # compute the losses
                loss_batch = compute_image_loss(loss_func,
                                                outputs.to(device),
                                                batch_labels.view(outputs.shape).to(device),
                                                device)
                # record the validation loss of a mini-batch
                valid_loss  += loss_batch.data
                denominator = ii

                y_true.append(batch_labels)
                y_pred.append(outputs.clone())
                features.append(_feature)
        valid_loss = valid_loss / (denominator + 1)
    return valid_loss,y_pred,y_true,features

def simple_FCNN_train_and_validation(
        model_to_train,
        f_name,
        loss_func,
        optimizer,
        image_resize        = 128,
        output_activation   = 'softmax',
        device              = 'cpu',
        batch_size          = 8,
        n_epochs            = int(3e3),
        print_train         = True,
        patience            = 5,
        train_root          = '',
        valid_root          = '',
        n_noise             = 0,
        noise_level         = None,
        tol                 = 1e-4,
        rotation            = True,
        gitter_color        = False,
        ):
    """
    This function is to train a new CNN model on clear images
    
    The training and validation processes should be modified accordingly if 
    new modules (i.e., a secondary network) are added to the model
    
    Arguments
    ---------------
    model_to_train:torch.nn.Module, a nn.Module class
    f_name:string, the name of the model that is to be trained
    loss_func:torch.nn.modules.loss, loss function
    optimizer:torch.optim, optimizer
    image_resize:int, default = 128, the number of pixels per axis for the image
        to be resized to
    output_activation: str
    device:string or torch.device, default = "cpu", where to train model
    batch_size:int, default = 8, batch size
    n_epochs:int, default = int(3e3), the maximum number of epochs for training
    print_train:bool, default = True, whether to show verbose information
    patience:int, default = 5, the number of epochs the model is continuely trained
        when the validation loss does not change
    train_root:string, default = '', the directory of data for training
    valid_root:string, default = '', the directory of data for validation
    tol: float, default = 1e-4
    rotation: bool, default = True,
    gitter_color: bool, default = False,
    
    Output
    -----------------
    model_to_train:torch.nn.Module, a nn.Module class
    losses: list of float, losses from the training and validation steps
    """
    augmentations = {
            'train':simple_augmentations(image_resize,noise_level = noise_level,rotation = rotation,gitter_color = gitter_color),
            'valid':simple_augmentations(image_resize,noise_level = noise_level,rotation = rotation,gitter_color = gitter_color),
        }
    
    train_loader        = data_loader(
            train_root,
            augmentations   = augmentations['train'],
            batch_size      = batch_size,
            )
    valid_loader        = data_loader(
            valid_root,
            augmentations   = augmentations['valid'],
            batch_size      = batch_size,
            )
    
    model_to_train.to(device)
    model_parameters    = filter(lambda p: p.requires_grad, model_to_train.parameters())
    if print_train:
        params          = sum([np.prod(p.size()) for p in model_parameters])
        print(f'total params: {params:d}')
    
    best_valid_loss     = np.inf
    losses              = []
    count               = 0
    for idx_epoch in range(n_epochs):
        # train
        print('\ntraining ...')
        _               = simple_FCNN_train_loop(
        net                 = model_to_train,
        loss_func           = loss_func,
        optimizer           = optimizer,
        dataloader          = train_loader,
        device              = device,
        output_activation   = output_activation,
        idx_epoch           = idx_epoch,
        print_train         = print_train,
        n_noise             = n_noise,
        )
        print('\nvalidating ...')
        valid_loss,y_pred,y_true,features = simple_FCNN_validation_loop(
        net                 = model_to_train,
        loss_func           = loss_func,
        dataloader          = valid_loader,
        device              = device,
        output_activation   = output_activation,
        )
        y_pred = torch.cat(y_pred)
        y_true = torch.cat(y_true)
        # if output_activation == 'softmax':
        #     answers = torch.argmax(y_true,1)
        #     binarized_y_pred = torch.argmax(y_pred,1)
        # elif output_activation == 'sigmoid':
        #     binarized_y_pred = y_pred >= 0.5
        #     binarized_y_pred = binarized_y_pred.float()
        # else:
        #     raise NotImplementedError
        # score = metrics.matthews_corrcoef(answers.detach().cpu().numpy(),
        #                                   binarized_y_pred.detach().cpu().numpy())
        score = metrics.roc_auc_score(y_true.detach().cpu(),y_pred.detach().cpu())
        
        
        # determine termination of training
        temp = valid_loss.cpu().clone().detach().type(torch.float64)
        if np.logical_and(temp < best_valid_loss, # if the new loss is lower than before
                          np.abs(best_valid_loss - temp) >= tol, # if the change is greater than tolerate
                          ):
            best_valid_loss = valid_loss.cpu().clone().detach().type(torch.float64)
            torch.save(model_to_train.state_dict(),f_name)# why do i need state_dict()?
            count = 0
        else:
            count += 1
        #     model_to_train.load_state_dict(torch.load(f_name,map_location = device))
        losses.append(best_valid_loss)
        _idx = np.random.choice(len(y_pred),size = 1)
        print(f'epoch {idx_epoch + 1}, loss = {valid_loss:6f},score = {score:.4f}, count = {count}')
        print(y_true[_idx])
        print(y_pred[_idx])
        if count >= patience:
            break
    return model_to_train,losses

def compute_image_loss(image_loss_func,image_category,labels,device):
    """
    
    """
    if "Binary Cross Entropy" in image_loss_func.__doc__:
        labels = labels.float()
        image_loss = image_loss_func(image_category.to(device),
                                     labels.view(image_category.shape).to(device)
                                     )
    elif "negative log likelihood loss" in image_loss_func.__doc__:
        labels = labels.long()
        image_loss = image_loss_func(torch.log(image_category).to(device),
                                     labels[:,-1].to(device))
    return image_loss

def create_meta_labels(image_category,labels,output_activation,idx_noise,device):
    """
    
    """
    with torch.no_grad():
        image_category  = image_category.to(device)
        labels          = labels.to(device)
        if output_activation == 'softmax':
            labels                      = torch.argmax(labels,1)
            binarized_image_category    = torch.argmax(image_category,1)
        elif output_activation == 'sigmoid':
            binarized_image_category    = image_category >= 0.5
            binarized_image_category    = binarized_image_category.float()
        else:
            raise NotImplementedError
        
        meta_labels = binarized_image_category == labels
        meta_labels = meta_labels.float()
        if len(idx_noise) > 0:
            meta_labels[idx_noise] = 0.5
        # meta_labels = torch.stack([meta_labels,1 - meta_labels]).T
        return meta_labels

def compute_meta_loss(meta_loss_func,meta_output,meta_labels,device,):
    """
    
    """
    if "Binary Cross Entropy" in meta_loss_func.__doc__:
        meta_labels = meta_labels.float()
        meta_labels = torch.stack([1 - meta_labels,meta_labels]).T
        meta_loss = meta_loss_func(meta_output.to(device),
                                   meta_labels.detach().to(device))
    elif "negative log likelihood loss" in meta_loss_func.__doc__:
        meta_labels = meta_labels.long()
        meta_loss = meta_loss_func(torch.log(meta_output).to(device),
                                    meta_labels.long().detach().to(device))
    return meta_loss

def reparameterize(mu,log_var,device,trick_type = 1):
        """
        generate a random distribution w.r.t. the mu and log_var from the latent space
        
        #eps = Variable(torch.FloatTensor(vector_size).normal_()).to(self.device)
        
        Inputs
        ---
        mu: torch.tensor
        log_var: torch.tensor
        device: str or torch.device
        trick_type: int,default = 1
        
        Output
        ---
        z: torch.tensor,
        """
        if trick_type == 1:
            vector_size = log_var.shape
            # independent noise from different dimensions
            dist        = torch.distributions.multivariate_normal.MultivariateNormal(
                                                    torch.zeros(vector_size[1]),
                                                    torch.eye(vector_size[1])
                                                    )
            # sample epsilon from a multivariate Gaussian distribution
            eps         = dist.sample((vector_size[0],)).to(device)
            # std = sqrt(exp(log_var))
            std         = torch.sqrt(torch.exp(log_var)).to(device)
            # z = mu + std * eps
            z = mu + std * eps
            return z
        elif trick_type == 2:
            # sample hidden representation from Q(z | x)
            std         = torch.sqrt(torch.exp(log_var))
            q           = torch.distributions.Normal(mu,std)
            z           = q.rsample()
            return z
        else:
            raise NotImplementedError
        
def kl_divergence(z, mu, log_var,formula = 1):
    """
    Inputs
    ---
    z: sampled space
    mu: mu space
    std: sqrt(exp(log_var))
    formula: int,
    
    Output
    ---
    kl
    """
    if formula == 1:
        KLD_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return KLD_loss
    elif formula == 2:
        std = torch.sqrt(torch.exp(log_var))
        # --------------------------
        # Monte carlo KL divergence
        # --------------------------
        # 1. define the first two probabilities (in this case Normal for both)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))#?why
        q = torch.distributions.Normal(mu, std)
    
        # 2. get the probabilities from the equation
        log_qzx = q.log_prob(z)
        log_pz = p.log_prob(z)
    
        # kl
        kl = (log_qzx - log_pz)
        kl = -0.5 * kl.sum(-1)
        return kl.mean()
###########################################
class GaussianNoise(nn.Module):
    """
    https://discuss.pytorch.org/t/where-is-the-noise-layer-in-pytorch/2887/4
    """
    """Gaussian noise regularizer.

    Args:
        sigma (float, optional): relative standard deviation used to generate the
            noise. Relative means that it will be multiplied by the magnitude of
            the value your are adding the noise to. This means that sigma can be
            the same regardless of the scale of the vector.
    """

    def __init__(self, 
                 sigma              = 0.1,
                 device             = 'cpu'):
        super(GaussianNoise,self).__init__()
        self.sigma              = sigma
        self.device             = device
        # self.noise              = torch.tensor(0.).float().to(self.device)

    def forward(self, x):
        with torch.no_grad():
            matrix_size         = x.shape
            # scale               = self.sigma * x.detach() # make sure the noise is not learnable
            # sampled_noise       = self.noise.repeat(*x.size()).normal_() * scale.float().to(self.device)
            # x                   = x.to(self.device) + sampled_noise.to(self.device)
            if self.sigma is not None:
                # generator       = torch.distributions.normal.Normal(0,self.sigma)
                # x               = x.to(self.device) + generator.sample(matrix_size).to(self.device)
                dist            = torch.distributions.multivariate_normal.MultivariateNormal(
                                                    loc = torch.zeros(matrix_size[1]),
                                                    covariance_matrix = torch.eye(matrix_size[1]) * self.sigma)
                x               = x.to(self.device) + dist.sample((matrix_size[0],)).to(self.device)
            return x.to(self.device)

class easy_model(nn.Module):
    """
    Models are not created equally
    Some pretrained models are composed by a {feature} and a {classifier} component
    thus, they are very easy to modify and transfer learning

    Inputs
    --------------------
    pretrained_model: nn.Module, pretrained model object

    Outputs
    --------------------
    out: torch.tensor, the pooled CNN features
    """
    def __init__(self,
                 pretrained_model,
                 in_shape = (1,3,128,128),
                 ):
        super(easy_model,self).__init__()
        torch.manual_seed(12345)
        self.in_features    = nn.AdaptiveAvgPool2d((1,1))(pretrained_model.features(torch.rand(*in_shape))).shape[1]
        avgpool             = nn.AdaptiveAvgPool2d((1,1))
        # print(f'feature dim = {self.in_features}')
        self.features       = nn.Sequential(pretrained_model.features,
                                            avgpool,)
    def forward(self,x,):
        out                 = torch.squeeze(torch.squeeze(self.features(x),3),2)
        return out

class resnet_model(nn.Module):
    """
    Models are not created equally
    Some pretrained models are composed by a {feature} and a {fc} component
    thus, they are very easy to modify and transfer learning

    Inputs
    --------------------
    pretrained_model: nn.Module, pretrained model object

    Outputs
    --------------------
    out: torch.tensor, the pooled CNN features
    """

    def __init__(self,
                 pretrained_model,
                 ):
        super(resnet_model,self).__init__()
        torch.manual_seed(12345)
        avgpool             = nn.AdaptiveAvgPool2d((1,1))
        self.in_features    = pretrained_model.fc.in_features
        res_net             = torch.nn.Sequential(*list(pretrained_model.children())[:-2])
        # print(f'feature dim = {self.in_features}')
        
        self.features       = nn.Sequential(res_net,
                                            avgpool)
        
    def forward(self,x):
        out                 = torch.squeeze(torch.squeeze(self.features(x),3),2)
        return out

class simple_FCNN(nn.Module):
    """
    A simple FCNN model
    Extract image features --> hidden representation --> image category
    
    Inputs
    ---
    pretrained_model_name: str,
    hidden_units: int
    hidden_activation: None or nn.Module, torch activation functions
    hidden_dropout: float, between 0 and 1
    output_units: int, 1 or 2
    output_activation: str, sigmoid or softmax
    in_shape: feature extractor input feature shape, default = (1,3,128,128)
    layer_type: str, currently only "linear" is implemented, default = 'linear'
    device: str or torch.device, default = 'cpu'
    
    Outputs
    ---
    image_category: torch.tensor
    hidden_representation: torch.tensor
    """
    def __init__(self,
                 pretrained_model_name,
                 hidden_units,
                 hidden_activation,
                 hidden_dropout,
                 output_units,
                 output_activation,
                 in_shape          = (1,3,128,128),
                 layer_type        = 'linear',
                 device            = 'cpu',
                 ):
        super(simple_FCNN,self).__init__()
        self.pretrained_model_name  = pretrained_model_name
        self.hidden_units           = hidden_units
        self.hidden_activation      = hidden_activation
        self.hidden_dropout         = hidden_dropout
        self.output_units           = output_units
        self.output_activation      = output_activation
        self.layer_type             = layer_type
        self.device                 = device
        
        torch.manual_seed(12345)
        self.pretrained_model       = candidates(pretrained_model_name)
        # freeze the pretrained model
        for params in self.pretrained_model.parameters():
            params.requires_grad    = False
        # get the dimensionof the CNN features
        if define_type(self.pretrained_model_name) == 'simple':
            self.in_features        = nn.AdaptiveAvgPool2d((1,1))(self.pretrained_model.features(torch.rand(*in_shape))).shape[1]
            self.feature_extractor  = easy_model(pretrained_model = self.pretrained_model,)
        elif define_type(self.pretrained_model_name) == 'resnet':
            self.in_features        = self.pretrained_model.fc.in_features
            self.feature_extractor  = resnet_model(pretrained_model = self.pretrained_model,)
        # hidden layer
        self.hidden_layer = internal_connection(
                                        layer_type         = self.layer_type,
                                        input_units         = self.in_features,
                                        output_units        = self.hidden_units,
                                        output_activation   = self.hidden_activation,
                                        output_dropout      = self.hidden_dropout,
                                        device              = self.device,
                                        ).to(device)
        # output layer
        self.output_layer = nn.Sequential(
                                        nn.Linear(self.hidden_units,
                                                  self.output_units),
                                        output_act_func_dict[self.output_activation]
                                        ).to(device)
    def forward(self,x):
        # extract the CNN features
        CNN_features = self.feature_extractor(x)
        # hidden layer
        hidden_representation = self.hidden_layer(CNN_features)
        # image category
        image_category = self.output_layer(hidden_representation)
        return image_category,hidden_representation

class simple_metacognitve_network(nn.Module):
    """
    hidden_units: int
    hidden_activation: nn.Module, Pytorch available activation functions
    hidden_dropout: float, between [0,1)
    output_units: int, being 1 or 2
    output_activation, nn.Module, being "softmax" only
    layer_type: so far , only linear connection is implemented
    device: str or torch.device
    internal_noise: None or float, > 0, noise added to the image representation
    """
    
    def __init__(self,
                 hidden_units,
                 hidden_activation,
                 hidden_dropout,
                 output_units,
                 output_activation,
                 layer_type = 'llinear', # TODO
                 device = 'cpu',
                 internal_noise = 0,
                 ):
        super(simple_metacognitve_network,self).__init__()
        
        self.hidden_units                   = hidden_units
        self.hidden_activation              = hidden_activation
        self.hidden_dropout                 = hidden_dropout
        self.output_units                   = output_units
        self.output_activation              = output_activation
        self.layer_type                     = layer_type
        self.device                         = device
        self.internal_noise                 = internal_noise
        
        torch.manual_seed(12345)
        self.Gaussian_noise_layer           = GaussianNoise(
                                                    self.internal_noise,
                                                    self.device,
                                                    ).to(self.device)
        self.hidden_layer                   = internal_connection(
                                                    layer_type          = self.layer_type,
                                                    input_units         = self.hidden_units,
                                                    output_units        = self.hidden_units,
                                                    output_dropout      = self.hidden_dropout,
                                                    output_activation   = self.hidden_activation,
                                                    device              = self.device,
                                                    ).to(device)
        self.output_layer                   = nn.Sequential(
                                                    nn.Linear(self.hidden_units,
                                                              self.output_units),
                                                    nn.Softmax(dim = 1)
                                                    ).to(device)
    def forward(self,x):
        if self.internal_noise is not None:
            x                   = self.Gaussian_noise_layer(x)
        hidden_representation   = self.hidden_layer(x)
        meta_output             = self.output_layer(hidden_representation)
        return hidden_representation,meta_output

class variational_FCNN(nn.Module):
    """
    # NOT WORKING
    Build a metacognitive network with noise added to the first hidden layer
    Input image --> CNN layer --> hidden representation --> mu -------
                                           |                          |--> image category
                                           ---------> log_var---------
    Inputs
    ---
    pretrained_model_name: str
    hidden_units: int
    hidden_activation: nn.Module, Pytorch available activation functions
    hidden_dropout: float, between [0,1)
    output_units: int, being 1 or 2
    output_activation, nn.Module, being "sigmoid" or "softmax"
    latent_units: int
    layer_type: so far , only linear connection is implemented
    latent_activation: str
    latent_dropout: float, between [0,1)
    second_output_units: int, default = 2 for softmax because this is what is implemented so far
    device: str or torch.device
    in_shape: tuple of 4 int
    
    Output
    ---
    category: torch.tensor, prediction of the image categories
    hidden_representation: torch.tensor, hidden representation of the images
    mu: torch.tensor, the mu output
    log_var: torch.tensor, the log_var output
    """
    def __init__(self,
                 pretrained_model_name,
                 hidden_units,
                 hidden_activation,
                 hidden_dropout,
                 output_units,
                 output_activation,
                 latent_units,
                 latent_activation,
                 latent_dropout,
                 in_shape = (1,3,128,128),
                 layer_type = 'linear',
                 device = 'cpu',
                 ):
        super(variational_FCNN,self).__init__()
        
        self.pretrained_model_name          = pretrained_model_name
        self.hidden_units                   = hidden_units
        self.hidden_activation              = hidden_activation
        self.hidden_dropout                 = hidden_dropout
        self.output_units                   = output_units
        self.output_activation              = output_activation
        self.latent_units                   = latent_units
        self.layer_type                     = layer_type
        self.latent_activation              = latent_activation
        self.latent_dropout                 = latent_dropout
        self.device                         = device
        
        torch.manual_seed(12345)
        self.pretrained_model               = candidates(pretrained_model_name)
        # freeze the pretrained model
        for params in self.pretrained_model.parameters():
            params.requires_grad            = False
        # get the dimensionof the CNN features
        if define_type(self.pretrained_model_name) == 'simple':
            self.in_features                = nn.AdaptiveAvgPool2d((1,1))(
                                                self.pretrained_model.features(
                                                        torch.rand(*in_shape))).shape[1]
            self.feature_extractor          = easy_model(pretrained_model = self.pretrained_model,)
        elif define_type(self.pretrained_model_name) == 'resnet':
            self.in_features                = self.pretrained_model.fc.in_features
            self.feature_extractor          = resnet_model(pretrained_model = self.pretrained_model,)
        # hidden layer
        self.hidden_layer                   = internal_connection(
                                                layer_type          = self.layer_type,
                                                input_units         = self.in_features,
                                                output_units        = self.hidden_units,
                                                output_dropout      = self.hidden_dropout,
                                                output_activation   = self.hidden_activation,
                                                device              = self.device,
                                                ).to(device)
        # the mu layer
        self.mu_layer                       = internal_connection(
                                                layer_type          = self.layer_type,
                                                input_units         = self.hidden_units,
                                                output_units        = self.output_units,
                                                output_activation   = nn.Sigmoid(),
                                                output_dropout      = self.hidden_dropout,
                                                device              = self.device,
                                                ).to(device)
        # the log_var layer
        self.log_var_layer                  = internal_connection(
                                                layer_type          = self.layer_type,
                                                input_units         = self.hidden_units,
                                                output_units        = self.output_units,
                                                output_activation   = nn.Sigmoid(),
                                                output_dropout      = self.hidden_dropout,
                                                device              = self.device,
                                                ).to(device)
        # z_layer is computed during forward passing and it will be the output categories
        # ...
        # # the output layer of the metacognitive network
        # self.output_layer                   = nn.Sequential(
        #                                         nn.ReLU(),
        #                                         nn.Linear(self.hidden_units,
        #                                                   self.hidden_units,),
        #                                         nn.ReLU(),
        #                                         nn.Linear(self.hidden_units,
        #                                                   self.output_units),
        #                                         output_act_func_dict[self.output_activation],
        #                                         ).to(device)
    
    def forward(self,x):
        # extract the CNN features
        CNN_features            = self.feature_extractor(x)
        # hidden representation
        hidden_representation   = self.hidden_layer(CNN_features)
        # compute the mu
        mu                      = self.mu_layer(hidden_representation)
        # compuate log_var
        log_var                 = self.log_var_layer(hidden_representation)
        # reparameterize trick + hidden activation function
        z                       = reparameterize(mu, log_var, self.device,trick_type = 1)
        # predictions
        image_category          = output_act_func_dict[self.output_activation](z)
        return image_category,hidden_representation ,mu,log_var,z

class noise_metacognitive_network(nn.Module):
    """
    Build a metacognitive network with noise added to the first hidden layer
    Input image --> CNN layer --> hidden layer --> output layer (image category)
                      noise-->|
                              --> hidden layer --> meta-layer (confidence)
    Inputs
    ---
    pretrained_model_name: str
    hidden_units: int
    hidden_activation: nn.Module, Pytorch available activation functions
    hidden_dropout: float, between [0,1)
    output_units: int, being 1 or 2
    output_activation, nn.Module, being "sigmoid" or "softmax"
    latent_units: int
    layer_type: so far , only linear connection is implemented
    latent_activation: str
    latent_dropout: float, between [0,1)
    second_output_units: int, default = 2 for softmax because this is what is implemented so far
    device: str or torch.device
    internal_noise: float, > 0, the noise injected to inbetween the first and second hidden layer
    in_shape: tuple of 4 int
    
    Output
    ---
    category: torch.tensor, prediction of the image categories
    hidden_representation: torch.tensor, hidden representation of the images
    meta_prediction: torch.tensor, prediction of the metacognitive network
    """
    def __init__(self,
                 pretrained_model_name      = 'vgg19',
                 hidden_units               = 2,
                 hidden_activation          = nn.ReLU,
                 hidden_dropout             = 0.,
                 output_units               = 2,
                 output_activation          = nn.Softmax(dim = 1),
                 latent_units               = 2,
                 layer_type                 = 'linear', 
                 latent_activation          = 'selu',
                 latent_dropout             = 0.,
                 second_output_units        = 2,
                 device                     = 'cpu',
                 internal_noise             = .1,
                 in_shape                   = (1,3,128,128)
                 ):
        super(noise_metacognitive_network,self).__init__()
        self.pretrained_model_name          = pretrained_model_name
        self.hidden_units                   = hidden_units
        self.hidden_activation              = hidden_activation
        self.hidden_dropout                 = hidden_dropout
        self.output_units                   = output_units
        self.output_activation              = output_activation
        self.latent_units                   = latent_units
        self.layer_type                     = layer_type
        self.latent_activation              = latent_activation
        self.latent_dropout                 = latent_dropout
        self.second_output_units            = second_output_units
        self.device                         = device
        self.internal_noise                 = internal_noise
        
        torch.manual_seed(12345)
        self.pretrained_model               = candidates(pretrained_model_name)
        # freeze the pretrained model
        for params in self.pretrained_model.parameters():
            params.requires_grad            = False
        # get the dimensionof the CNN features
        if define_type(self.pretrained_model_name) == 'simple':
            self.in_features                = nn.AdaptiveAvgPool2d((1,1))(
                                                self.pretrained_model.features(
                                                        torch.rand(*in_shape))).shape[1]
            self.feature_extractor          = easy_model(pretrained_model = self.pretrained_model,)
        elif define_type(self.pretrained_model_name) == 'resnet':
            self.in_features                = self.pretrained_model.fc.in_features
            self.feature_extractor          = resnet_model(pretrained_model = self.pretrained_model,)
        # hidden layer
        self.hidden_layer                   = internal_connection(
                                                layer_type          = self.layer_type,
                                                input_units         = self.in_features,
                                                output_units        = self.hidden_units,
                                                output_dropout      = self.hidden_dropout,
                                                output_activation   = self.hidden_activation,
                                                device              = self.device,
                                                ).to(device)
        # predict layer for images
        self.output_layer                   = nn.Sequential(
                                                nn.Linear(self.hidden_units,self.output_units),
                                                output_act_func_dict[self.output_activation],
                                                ).to(device)
        # noise layer
        if self.internal_noise is not None:
            self.noise_layer                = GaussianNoise(
                                                    sigma           = self.internal_noise,
                                                    device          = self.device,
                                                    ).to(device)
        # secondary hidden layer
        self.secondary_hidden_layer         = internal_connection(
                                                layer_type          = self.layer_type,
                                                input_units         = self.hidden_units,
                                                output_units        = self.latent_units,
                                                output_dropout      = self.latent_dropout,
                                                output_activation   = self.latent_activation,
                                                device              = self.device,
                                                ).to(device)
        # meta-prediction layer
        self.meta_output_layer              = nn.Sequential(
                                                nn.Linear(self.latent_units,self.second_output_units,),
                                                nn.Softmax(dim = 1),
                                                ).to(device)
        # softmax layer
        # self.softmax                        = nn.Softmax(dim = 1)
    def forward(self,x):
        # extract the CNN features
        CNN_features                    = self.feature_extractor(x)
        # get hidden representations
        hidden_representation           = self.hidden_layer(CNN_features)
        # predict the image category
        category                        = self.output_layer(hidden_representation)
        if self.internal_noise is not None:
            # noise hidden representation
            noisy_hidden_representation     = self.noise_layer(hidden_representation)
            # secondary hidden layer
            secondary_hidden_representation = self.secondary_hidden_layer(noisy_hidden_representation)
        else:
            secondary_hidden_representation = self.secondary_hidden_layer(hidden_representation)
        # meta-prediction
        meta_prediction                 = self.meta_output_layer(secondary_hidden_representation)
        return category, hidden_representation, meta_prediction

class variational_metacognitive_network(nn.Module):
    """
    Build a variational metacognitive network
    Input image --> CNN layer --> hidden layer --> output layer
                              |
                              --> mu          |
                              |               |--> z layer --> meta-layer
                              --> log variance|
    Inputs
    ---
    pretrained_model_name: str
    hidden_units: int
    hidden_activation: nn.Module, Pytorch available activation functions
    hidden_dropout: float, between [0,1)
    output_units: int, being 1 or 2
    output_activation, nn.Module, being "sigmoid" or "softmax"
    layer_units: int
    layer_type: so far , only linear connection is implemented
    layer_activation: str
    layer_dropout: float, between [0,1)
    second_output_units: int, default = 2 for softmax because this is what is implemented so far
    device: str or torch.device
    internal_noise: float, > 0, the noise injected to inbetween the first and second hidden layer
    
    Output
    ---
    category: torch.tensor, prediction of the image categories
    hidden_representation: torch.tensor, hidden representation of the images
    mu: torch.tensor,
    log_var: torch.tensor
    z: torch.tensor, the sampled representation of the hidden
                            representation of the images
    meta_prediction: torch.tensor, prediction of the metacognitive network
    """
    def __init__(self,
                 pretrained_model_name      = 'vgg19',
                 hidden_units               = 2,
                 hidden_activation          = nn.ReLU,
                 hidden_dropout             = 0.,
                 output_units               = 2,
                 output_activation          = nn.Softmax(dim = 1),
                 latent_units               = 2,
                 layer_type                 = 'linear', # or RNN
                 latent_activation          = 'selu',
                 latent_dropout             = 0.,
                 second_output_units        = 2,
                 device                     = 'cpu',
                 in_shape                   = (1,3,128,128),
                 ):
        super(variational_metacognitive_network,self).__init__()
        self.pretrained_model_name          = pretrained_model_name
        self.hidden_units                   = hidden_units
        self.hidden_activation              = hidden_activation
        self.hidden_dropout                 = hidden_dropout
        self.output_units                   = output_units
        self.output_activation              = output_activation
        self.latent_units                   = latent_units
        self.layer_type                     = layer_type
        self.latent_activation              = latent_activation
        self.latent_dropout                 = latent_dropout
        self.second_output_units            = second_output_units
        self.device                         = device
        
        torch.manual_seed(12345)
        self.pretrained_model               = candidates(pretrained_model_name)
        # freeze the pretrained model
        for params in self.pretrained_model.parameters():
            params.requires_grad            = False
        # get the dimensionof the CNN features
        if define_type(self.pretrained_model_name) == 'simple':
            self.in_features                = nn.AdaptiveAvgPool2d((1,1))(
                                                self.pretrained_model.features(
                                                        torch.rand(*in_shape))).shape[1]
            self.feature_extractor          = easy_model(pretrained_model = self.pretrained_model,)
        elif define_type(self.pretrained_model_name) == 'resnet':
            self.in_features                = self.pretrained_model.fc.in_features
            self.feature_extractor          = resnet_model(pretrained_model = self.pretrained_model,)
        # hidden layer
        self.hidden_layer                   = internal_connection(
                                                layer_type          = self.layer_type,
                                                input_units         = self.in_features,
                                                output_units        = self.hidden_units,
                                                output_activation   = self.hidden_activation,
                                                output_dropout      = self.hidden_dropout,
                                                device              = self.device,
                                                ).to(device)
        # output layer of the image category
        self.output_layer                   = nn.Sequential(
                                                nn.Linear(self.hidden_units,self.output_units,),
                                                output_act_func_dict[self.output_activation],
                                                ).to(device)
        # the mu layer
        self.hidden_mu_layer                = internal_connection(
                                                layer_type          = self.layer_type,
                                                input_units         = self.hidden_units,
                                                output_units        = self.second_output_units,
                                                output_activation   = nn.Tanh(),
                                                output_dropout      = self.latent_dropout,
                                                device              = self.device,
                                                ).to(device)
        # the log_var layer
        self.hidden_log_var_layer           = internal_connection(
                                                layer_type          = self.layer_type,
                                                input_units         = self.hidden_units,
                                                output_units        = self.second_output_units,
                                                output_activation   = nn.Tanh(),
                                                output_dropout      = self.latent_dropout,
                                                device              = self.device,
                                                ).to(device)
        # the mu layer
        self.output_mu_layer                = internal_connection(
                                                layer_type          = self.layer_type,
                                                input_units         = self.output_units,
                                                output_units        = self.second_output_units,
                                                output_activation   = nn.Tanh(),
                                                output_dropout      = self.latent_dropout,
                                                device              = self.device,
                                                ).to(device)
        # the log_var layer
        self.output_log_var_layer           = internal_connection(
                                                layer_type          = self.layer_type,
                                                input_units         = self.output_units,
                                                output_units        = self.second_output_units,
                                                output_activation   = nn.Tanh(),
                                                output_dropout      = self.latent_dropout,
                                                device              = self.device,
                                                ).to(device)
        # z_layer is computed during forward passing
        # ......
        # the output layer of the metacognitive network
        # self.meta_output_layer               = nn.Sequential(
        #                                         nn.Linear(self.latent_units,
        #                                                   self.second_output_units),
        #                                         nn.Softmax(dim = 1)
        #                                         ).to(device)
        self.softmax = nn.Softmax(dim = 1)
    
    def forward(self,x):
        # extract the CNN features
        CNN_features                    = self.feature_extractor(x)
        # get hidden representations
        hidden_representation           = self.hidden_layer(CNN_features)
        # predict the image category
        category                        = self.output_layer(hidden_representation)
        # copmute mu and log_var from the classification hidden layer
        hidden_mu                       = self.hidden_mu_layer(hidden_representation)
        hidden_log_var                  = self.hidden_log_var_layer(hidden_representation)
        # reparameterization trick
        hidden_z                        = reparameterize(hidden_mu,hidden_log_var,
                                                         self.device,trick_type = 1)
        # compute the mu and log_var from the classification decision layer
        output_mu                       = self.output_mu_layer(category)
        output_log_var                  = self.output_log_var_layer(category)
        # reparameterization trick
        output_z                        = reparameterize(output_mu, output_log_var,
                                                         self.device,trick_type = 1)
        # prediction
        meta_prediction                 = self.softmax(hidden_z + output_z)
        return (category, hidden_representation, meta_prediction,
                hidden_mu,hidden_log_var,hidden_z,
                output_mu,output_log_var,output_z,)
########################################################################################################
##################  type 2 signal detection theory function ############################################
################## http://www.columbia.edu/~bsm2105/type2sdt/trials2counts.py ##########################
################## http://www.columbia.edu/~bsm2105/type2sdt/fit_meta_d_MLE.py #########################
"""
[translated by alan lee from trials2counts.m by Maniscalco & Lau (2012)]
[requires numpy-1.16.4, scipy-1.3.0, or later versions]
[comments below are copied and pasted from trials2counts.m]

function [nR_S1, nR_S2] = trials2counts(stimID, response, rating, nRatings, padCells, padAmount)

% Given data from an experiment where an observer discriminates between two
% stimulus alternatives on every trial and provides confidence ratings,
% converts trial by trial experimental information for N trials into response 
% counts.
%
% INPUTS
% stimID:   1xN vector. stimID(i) = 0 --> stimulus on i'th trial was S1.
%                       stimID(i) = 1 --> stimulus on i'th trial was S2.
%
% response: 1xN vector. response(i) = 0 --> response on i'th trial was "S1".
%                       response(i) = 1 --> response on i'th trial was "S2".
%
% rating:   1xN vector. rating(i) = X --> rating on i'th trial was X.
%                       X must be in the range 1 <= X <= nRatings.
%
% N.B. all trials where stimID is not 0 or 1, response is not 0 or 1, or
% rating is not in the range [1, nRatings], are omitted from the response
% count.
%
% nRatings: total # of available subjective ratings available for the
%           subject. e.g. if subject can rate confidence on a scale of 1-4,
%           then nRatings = 4
%
% optional inputs
%
% padCells: if set to 1, each response count in the output has the value of
%           padAmount added to it. Padding cells is desirable if trial counts 
%           of 0 interfere with model fitting.
%           if set to 0, trial counts are not manipulated and 0s may be
%           present in the response count output.
%           default value for padCells is 0.
%
% padAmount: the value to add to each response count if padCells is set to 1.
%            default value is 1/(2*nRatings)
%
%
% OUTPUTS
% nR_S1, nR_S2
% these are vectors containing the total number of responses in
% each response category, conditional on presentation of S1 and S2.
%
% e.g. if nR_S1 = [100 50 20 10 5 1], then when stimulus S1 was
% presented, the subject had the following response counts:
% responded S1, rating=3 : 100 times
% responded S1, rating=2 : 50 times
% responded S1, rating=1 : 20 times
% responded S2, rating=1 : 10 times
% responded S2, rating=2 : 5 times
% responded S2, rating=3 : 1 time
%
% The ordering of response / rating counts for S2 should be the same as it
% is for S1. e.g. if nR_S2 = [3 7 8 12 27 89], then when stimulus S2 was
% presented, the subject had the following response counts:
% responded S1, rating=3 : 3 times
% responded S1, rating=2 : 7 times
% responded S1, rating=1 : 8 times
% responded S2, rating=1 : 12 times
% responded S2, rating=2 : 27 times
% responded S2, rating=3 : 89 times
"""

def trials2counts(stimID, response, rating, nRatings, padCells = 0, padAmount = None):

    ''' sort inputs '''
    # check for valid inputs
    if not ( len(stimID) == len(response)) and (len(stimID) == len(rating)):
        raise('stimID, response, and rating input vectors must have the same lengths')
    
    ''' filter bad trials '''
    tempstim = []
    tempresp = []
    tempratg = []
    for s,rp,rt in zip(stimID,response,rating):
        if (s == 0 or s == 1) and (rp == 0 or rp == 1) and (rt >=1 and rt <= nRatings):
            tempstim.append(s)
            tempresp.append(rp)
            tempratg.append(rt)
    stimID = tempstim
    response = tempresp
    rating = tempratg
    
    ''' set input defaults '''
    if padAmount == None:
        padAmount = 1/(2*nRatings)
    
    ''' compute response counts '''
    nR_S1 = []
    nR_S2 = []

    # S1 responses
    for r in range(nRatings,0,-1):
        cs1, cs2 = 0,0
        for s,rp,rt in zip(stimID, response, rating):
            if s==0 and rp==0 and rt==r:
                cs1 += 1
            if s==1 and rp==0 and rt==r:
                cs2 += 1
        nR_S1.append(cs1)
        nR_S2.append(cs2)
    
    # S2 responses
    for r in range(1,nRatings+1,1):
        cs1, cs2 = 0,0
        for s,rp,rt in zip(stimID, response, rating):
            if s==0 and rp==1 and rt==r:
                cs1 += 1
            if s==1 and rp==1 and rt==r:
                cs2 += 1
        nR_S1.append(cs1)
        nR_S2.append(cs2)
    
    # pad response counts to avoid zeros
    if padCells:
        nR_S1 = [n+padAmount for n in nR_S1]
        nR_S2 = [n+padAmount for n in nR_S2]
        
    return nR_S1, nR_S2
  
"""
Created on Fri Jul 12 14:13:24 2019

[translated from fit_meta_d_MLE.m by Maniscalco & Lau (2012) by alan lee]
[requires numpy-1.13.3, scipy-1.1.0, or later versions]
[comments below are copied and pasted from fit_meta_d_MLE.m]

function fit = fit_meta_d_MLE(nR_S1, nR_S2, s, fncdf, fninv)

% fit = fit_meta_d_MLE(nR_S1, nR_S2, s, fncdf, fninv)
%
% Given data from an experiment where an observer discriminates between two
% stimulus alternatives on every trial and provides confidence ratings,
% provides a type 2 SDT analysis of the data.
%
% INPUTS
%
% * nR_S1, nR_S2
% these are vectors containing the total number of responses in
% each response category, conditional on presentation of S1 and S2.
%
% e.g. if nR_S1 = [100 50 20 10 5 1], then when stimulus S1 was
% presented, the subject had the following response counts:
% responded S1, rating=3 : 100 times
% responded S1, rating=2 : 50 times
% responded S1, rating=1 : 20 times
% responded S2, rating=1 : 10 times
% responded S2, rating=2 : 5 times
% responded S2, rating=3 : 1 time
%
% The ordering of response / rating counts for S2 should be the same as it
% is for S1. e.g. if nR_S2 = [3 7 8 12 27 89], then when stimulus S2 was
% presented, the subject had the following response counts:
% responded S1, rating=3 : 3 times
% responded S1, rating=2 : 7 times
% responded S1, rating=1 : 8 times
% responded S2, rating=1 : 12 times
% responded S2, rating=2 : 27 times
% responded S2, rating=3 : 89 times
%
% N.B. if nR_S1 or nR_S2 contain zeros, this may interfere with estimation of
% meta-d'.
%
% Some options for dealing with response cell counts containing zeros are:
% 
% (1) Add a small adjustment factor, e.g. adj_f = 1/(length(nR_S1), to each 
% input vector:
% 
% adj_f = 1/length(nR_S1);
% nR_S1_adj = nR_S1 + adj_f;
% nR_S2_adj = nR_S2 + adj_f;
% 
% This is a generalization of the correction for similar estimation issues of
% type 1 d' as recommended in
% 
% Hautus, M. J. (1995). Corrections for extreme proportions and their biasing 
%     effects on estimated values of d'. Behavior Research Methods, Instruments, 
%     & Computers, 27, 46-51.
%     
% When using this correction method, it is recommended to add the adjustment 
% factor to ALL data for all subjects, even for those subjects whose data is 
% not in need of such correction, in order to avoid biases in the analysis 
% (cf Snodgrass & Corwin, 1988).
% 
% (2) Collapse across rating categories.
% 
% e.g. if your data set has 4 possible confidence ratings such that length(nR_S1)==8,
% defining new input vectors
% 
% nR_S1_new = [sum(nR_S1(1:2)), sum(nR_S1(3:4)), sum(nR_S1(5:6)), sum(nR_S1(7:8))];
% nR_S2_new = [sum(nR_S2(1:2)), sum(nR_S2(3:4)), sum(nR_S2(5:6)), sum(nR_S2(7:8))];
% 
% might be sufficient to eliminate zeros from the input without using an adjustment.
%
% * s
% this is the ratio of standard deviations for type 1 distributions, i.e.
%
% s = sd(S1) / sd(S2)
%
% if not specified, s is set to a default value of 1.
% For most purposes, we recommend setting s = 1. 
% See http://www.columbia.edu/~bsm2105/type2sdt for further discussion.
%
% * fncdf
% a function handle for the CDF of the type 1 distribution.
% if not specified, fncdf defaults to @normcdf (i.e. CDF for normal
% distribution)
%
% * fninv
% a function handle for the inverse CDF of the type 1 distribution.
% if not specified, fninv defaults to @norminv
%
% OUTPUT
%
% Output is packaged in the struct "fit." 
% In the following, let S1 and S2 represent the distributions of evidence 
% generated by stimulus classes S1 and S2.
% Then the fields of "fit" are as follows:
% 
% fit.da        = mean(S2) - mean(S1), in room-mean-square(sd(S1),sd(S2)) units
% fit.s         = sd(S1) / sd(S2)
% fit.meta_da   = meta-d' in RMS units
% fit.M_diff    = meta_da - da
% fit.M_ratio   = meta_da / da
% fit.meta_ca   = type 1 criterion for meta-d' fit, RMS units
% fit.t2ca_rS1  = type 2 criteria of "S1" responses for meta-d' fit, RMS units
% fit.t2ca_rS2  = type 2 criteria of "S2" responses for meta-d' fit, RMS units
%
% fit.S1units   = contains same parameters in sd(S1) units.
%                 these may be of use since the data-fitting is conducted  
%                 using parameters specified in sd(S1) units.
% 
% fit.logL          = log likelihood of the data fit
%
% fit.est_HR2_rS1  = estimated (from meta-d' fit) type 2 hit rates for S1 responses
% fit.obs_HR2_rS1  = actual type 2 hit rates for S1 responses
% fit.est_FAR2_rS1 = estimated type 2 false alarm rates for S1 responses
% fit.obs_FAR2_rS1 = actual type 2 false alarm rates for S1 responses
% 
% fit.est_HR2_rS2  = estimated type 2 hit rates for S2 responses
% fit.obs_HR2_rS2  = actual type 2 hit rates for S2 responses
% fit.est_FAR2_rS2 = estimated type 2 false alarm rates for S2 responses
% fit.obs_FAR2_rS2 = actual type 2 false alarm rates for S2 responses
%
% If there are N ratings, then there will be N-1 type 2 hit rates and false
% alarm rates. 

% 2019/07/12 - translated to python3 by alan lee
                [requires numpy-1.13.3, scipy-1.1.0, or later versions]
% 2015/07/23 - fixed bug for output fit.meta_ca and fit.S1units.meta_c1. 
%            - added comments to help section as well as a warning output 
%              for nR_S1 or nR_S2 inputs containing zeros
% 2014/10/14 - updated discussion of "s" input in the help section above.
% 2010/09/07 - created
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import Bounds, LinearConstraint, minimize, SR1

# returns negative log-likelihood of parameters given experimental data
# parameters[0] = meta d'
# parameters[1:end] = type-2 criteria locations
def fit_meta_d_logL(parameters,inputObj):
    meta_d1 = parameters[0]
    t2c1    = parameters[1:]
    nR_S1, nR_S2, nRatings, d1, t1c1, s, constant_criterion, fncdf, fninv = inputObj

    # define mean and SD of S1 and S2 distributions
    S1mu = -meta_d1/2
    S1sd = 1
    S2mu = meta_d1/2
    S2sd = S1sd/s

    # adjust so that the type 1 criterion is set at 0
    # (this is just to work with optimization toolbox constraints...
    #  to simplify defining the upper and lower bounds of type 2 criteria)
    S1mu = S1mu - eval(constant_criterion)
    S2mu = S2mu - eval(constant_criterion)

    t1c1 = 0

    # set up MLE analysis
    # get type 2 response counts
    # S1 responses
    nC_rS1 = [nR_S1[i] for i in range(nRatings)]
    nI_rS1 = [nR_S2[i] for i in range(nRatings)]
    # S2 responses
    nC_rS2 = [nR_S2[i+nRatings] for i in range(nRatings)]
    nI_rS2 = [nR_S1[i+nRatings] for i in range(nRatings)]

    # get type 2 probabilities
    C_area_rS1 = fncdf(t1c1,S1mu,S1sd)
    I_area_rS1 = fncdf(t1c1,S2mu,S2sd)
    
    C_area_rS2 = 1-fncdf(t1c1,S2mu,S2sd)
    I_area_rS2 = 1-fncdf(t1c1,S1mu,S1sd)
    
    t2c1x = [-np.inf]
    t2c1x.extend(t2c1[0:(nRatings-1)])
    t2c1x.append(t1c1)
    t2c1x.extend(t2c1[(nRatings-1):])
    t2c1x.append(np.inf)

    prC_rS1 = [( fncdf(t2c1x[i+1],S1mu,S1sd) - fncdf(t2c1x[i],S1mu,S1sd) ) / C_area_rS1 for i in range(nRatings)]
    prI_rS1 = [( fncdf(t2c1x[i+1],S2mu,S2sd) - fncdf(t2c1x[i],S2mu,S2sd) ) / I_area_rS1 for i in range(nRatings)]

    prC_rS2 = [( (1-fncdf(t2c1x[nRatings+i],S2mu,S2sd)) - (1-fncdf(t2c1x[nRatings+i+1],S2mu,S2sd)) ) / C_area_rS2 for i in range(nRatings)]
    prI_rS2 = [( (1-fncdf(t2c1x[nRatings+i],S1mu,S1sd)) - (1-fncdf(t2c1x[nRatings+i+1],S1mu,S1sd)) ) / I_area_rS2 for i in range(nRatings)]

    # calculate logL
    logL = np.sum([
            nC_rS1[i]*np.log(prC_rS1[i]) \
            + nI_rS1[i]*np.log(prI_rS1[i]) \
            + nC_rS2[i]*np.log(prC_rS2[i]) \
            + nI_rS2[i]*np.log(prI_rS2[i]) for i in range(nRatings)])
    
    if np.isinf(logL) or np.isnan(logL):
#        logL=-np.inf
        logL=-1e+300 # returning "-inf" may cause optimize.minimize() to fail
    return -logL


def fit_meta_d_MLE(nR_S1, nR_S2, s = 1, fncdf = norm.cdf, fninv = norm.ppf):

    # check inputs
    if (len(nR_S1) % 2)!=0: 
        raise('input arrays must have an even number of elements')
    if len(nR_S1)!=len(nR_S2):
        raise('input arrays must have the same number of elements')
    if any(np.array(nR_S1) == 0) or any(np.array(nR_S2) == 0):
        print(' ')
        print('WARNING!!')
        print('---------')
        print('Your inputs')
        print(' ')
        print('nR_S1:')
        print(nR_S1)
        print('nR_S2:')
        print(nR_S2)
        print(' ')
        print('contain zeros! This may interfere with proper estimation of meta-d''.')
        print('See ''help fit_meta_d_MLE'' for more information.')
        print(' ')
        print(' ')
    
    nRatings = int(len(nR_S1) / 2)  # number of ratings in the experiment
    nCriteria = int(2*nRatings - 1) # number criteria to be fitted
    
    """
    set up constraints for scipy.optimize.minimum()
    """
    # parameters
    # meta-d' - 1
    # t2c     - nCriteria-1
    # constrain type 2 criteria values,
    # such that t2c(i) is always <= t2c(i+1)
    # want t2c(i)   <= t2c(i+1) 
    # -->  t2c(i+1) >= t2c(i) + 1e-5 (i.e. very small deviation from equality) 
    # -->  t2c(i) - t2c(i+1) <= -1e-5 
    A = []
    ub = []
    lb = []
    for ii in range(nCriteria-2):
        tempArow = []
        tempArow.extend(np.zeros(ii+1))
        tempArow.extend([1, -1])
        tempArow.extend(np.zeros((nCriteria-2)-ii-1))
        A.append(tempArow)
        ub.append(-1e-5)
        lb.append(-np.inf)
        
    # lower bounds on parameters
    LB = []
    LB.append(-10.)                              # meta-d'
    LB.extend(-20*np.ones((nCriteria-1)//2))    # criteria lower than t1c
    LB.extend(np.zeros((nCriteria-1)//2))       # criteria higher than t1c
    
    # upper bounds on parameters
    UB = []
    UB.append(10.)                           # meta-d'
    UB.extend(np.zeros((nCriteria-1)//2))      # criteria lower than t1c
    UB.extend(20*np.ones((nCriteria-1)//2))    # criteria higher than t1c
    
    """
    prepare other inputs for scipy.optimize.minimum()
    """
    # select constant criterion type
    constant_criterion = 'meta_d1 * (t1c1 / d1)' # relative criterion
    
    # set up initial guess at parameter values
    ratingHR  = []
    ratingFAR = []
    for c in range(1,int(nRatings*2)):
        ratingHR.append(sum(nR_S2[c:]) / sum(nR_S2))
        ratingFAR.append(sum(nR_S1[c:]) / sum(nR_S1))
    
    # obtain index in the criteria array to mark Type I and Type II criteria
    t1_index = nRatings-1
    t2_index = list(set(list(range(0,2*nRatings-1))) - set([t1_index]))
    
    d1 = (1/s) * fninv( ratingHR[t1_index] ) - fninv( ratingFAR[t1_index] )
    meta_d1 = d1
    
    c1 = (-1/(1+s)) * ( fninv( ratingHR ) + fninv( ratingFAR ) )
    t1c1 = c1[t1_index]
    t2c1 = c1[t2_index]
    
    # initial values for the minimization function
    guess = [meta_d1]
    guess.extend(list(t2c1 - eval(constant_criterion)))
    
    # other inputs for the minimization function
    inputObj = [nR_S1, nR_S2, nRatings, d1, t1c1, s, constant_criterion, fncdf, fninv]        
    bounds = Bounds(LB,UB)
    linear_constraint = LinearConstraint(A,lb,ub)
    
    # minimization of negative log-likelihood
    results = minimize(fit_meta_d_logL, guess, args = (inputObj), method='trust-constr',
                       jac='2-point', hess=SR1(),
                       constraints = [linear_constraint],
                       options = {'verbose': 1}, bounds = bounds)
    
    # quickly process some of the output
    meta_d1 = results.x[0]
    t2c1    = results.x[1:] + eval(constant_criterion)
    logL    = -results.fun
    
    # data is fit, now to package it...
    # find observed t2FAR and t2HR 
    
    # I_nR and C_nR are rating trial counts for incorrect and correct trials
    # element i corresponds to # (in)correct w/ rating i
    I_nR_rS2 = nR_S1[nRatings:]
    I_nR_rS1 = list(np.flip(nR_S2[0:nRatings],axis=0))
    
    C_nR_rS2 = nR_S2[nRatings:];
    C_nR_rS1 = list(np.flip(nR_S1[0:nRatings],axis=0))
    
    obs_FAR2_rS2 = [sum( I_nR_rS2[(i+1):] ) / sum(I_nR_rS2) for i in range(nRatings-1)]
    obs_HR2_rS2 = [sum( C_nR_rS2[(i+1):] ) / sum(C_nR_rS2) for i in range(nRatings-1)]
    obs_FAR2_rS1 = [sum( I_nR_rS1[(i+1):] ) / sum(I_nR_rS1) for i in range(nRatings-1)]
    obs_HR2_rS1 = [sum( C_nR_rS1[(i+1):] ) / sum(C_nR_rS1) for i in range(nRatings-1)]
    
    # find estimated t2FAR and t2HR
    S1mu = -meta_d1/2
    S1sd = 1
    S2mu =  meta_d1/2
    S2sd = S1sd/s;
    
    mt1c1 = eval(constant_criterion)
    
    C_area_rS2 = 1-fncdf(mt1c1,S2mu,S2sd)
    I_area_rS2 = 1-fncdf(mt1c1,S1mu,S1sd)
    
    C_area_rS1 = fncdf(mt1c1,S1mu,S1sd)
    I_area_rS1 = fncdf(mt1c1,S2mu,S2sd)
    
    est_FAR2_rS2 = []
    est_HR2_rS2 = []
    
    est_FAR2_rS1 = []
    est_HR2_rS1 = []
    
    
    for i in range(nRatings-1):
        
        t2c1_lower = t2c1[(nRatings-1)-(i+1)]
        t2c1_upper = t2c1[(nRatings-1)+i]
            
        I_FAR_area_rS2 = 1-fncdf(t2c1_upper,S1mu,S1sd)
        C_HR_area_rS2  = 1-fncdf(t2c1_upper,S2mu,S2sd)
    
        I_FAR_area_rS1 = fncdf(t2c1_lower,S2mu,S2sd)
        C_HR_area_rS1  = fncdf(t2c1_lower,S1mu,S1sd)
    
        est_FAR2_rS2.append(I_FAR_area_rS2 / I_area_rS2)
        est_HR2_rS2.append(C_HR_area_rS2 / C_area_rS2)
        
        est_FAR2_rS1.append(I_FAR_area_rS1 / I_area_rS1)
        est_HR2_rS1.append(C_HR_area_rS1 / C_area_rS1)
    
    
    # package output
    fit = {}
    fit['da']       = np.sqrt(2/(1+s**2)) * s * d1
    
    fit['s']        = s
    
    fit['meta_da']  = np.sqrt(2/(1+s**2)) * s * meta_d1
    
    fit['M_diff']   = fit['meta_da'] - fit['da']
    
    fit['M_ratio']  = fit['meta_da'] / fit['da']
    
    mt1c1         = eval(constant_criterion)
    fit['meta_ca']  = ( np.sqrt(2)*s / np.sqrt(1+s**2) ) * mt1c1
    
    t2ca          = ( np.sqrt(2)*s / np.sqrt(1+s**2) ) * np.array(t2c1)
    fit['t2ca_rS1']     = t2ca[0:nRatings-1]
    fit['t2ca_rS2']     = t2ca[(nRatings-1):]
    
    fit['S1units'] = {}
    fit['S1units']['d1']        = d1
    fit['S1units']['meta_d1']   = meta_d1
    fit['S1units']['s']         = s
    fit['S1units']['meta_c1']   = mt1c1
    fit['S1units']['t2c1_rS1']  = t2c1[0:nRatings-1]
    fit['S1units']['t2c1_rS2']  = t2c1[(nRatings-1):]
    
    fit['logL']    = logL
    
    fit['est_HR2_rS1']  = est_HR2_rS1
    fit['obs_HR2_rS1']  = obs_HR2_rS1
    
    fit['est_FAR2_rS1'] = est_FAR2_rS1
    fit['obs_FAR2_rS1'] = obs_FAR2_rS1
    
    fit['est_HR2_rS2']  = est_HR2_rS2
    fit['obs_HR2_rS2']  = obs_HR2_rS2
    
    fit['est_FAR2_rS2'] = est_FAR2_rS2
    fit['obs_FAR2_rS2'] = obs_FAR2_rS2

    return fit
  ########################################################################################################
  ########################################################################################################
