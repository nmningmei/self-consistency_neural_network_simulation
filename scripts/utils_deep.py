#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 13:03:04 2022

@author: nmei

I didn't know spyder has a quick documentation feature

"""
from typing import List, Callable, Union, Any, TypeVar, Tuple, Optional
###############################################################################
Tensor = TypeVar('torch.tensor')
###############################################################################
import os,gc
from glob        import glob
from tqdm        import tqdm
from collections import OrderedDict
import pandas as pd
import numpy  as np

import torch
from torch          import nn
from torch.utils    import data
from torch.nn       import functional as F
from torch          import optim

from torchvision.datasets import ImageFolder
from torchvision          import transforms,datasets
from torchvision          import models as Tmodels

# from sklearn                 import metrics
# from sklearn.preprocessing   import StandardScaler
# from sklearn.svm             import LinearSVC,SVC
# from sklearn.decomposition   import PCA
# from sklearn.pipeline        import make_pipeline
# from sklearn.model_selection import StratifiedShuffleSplit,cross_validate,permutation_test_score
# from sklearn.calibration     import CalibratedClassifierCV
# from sklearn.linear_model    import LogisticRegression
# from sklearn.utils           import shuffle as sk_shuffle
# from sklearn.ensemble        import RandomForestClassifier
from sklearn.metrics         import roc_auc_score

def standard_dataset(generator:datasets,
                     train_valid_split:List,
                     unpack:dict,
                     ) -> Tuple:
    """
    A function for getting standard CV datasets from torchvision

    Parameters
    ----------
    generator : datasets object
        torchvision.datasets object being a generator.
    train_valid_split : List
        Containing two elements, the size of training and the size of the validation.
    unpack : dict
        DESCRIPTION.

    Raises
    ------
    NotImplementedError
        DESCRIPTION.

    Returns
    -------
    dataloaders : Tuple
        DESCRIPTION.

    """
    if train_valid_split is not None:
        torch.manual_seed(12345)
        train,valid  = data.random_split(generator,train_valid_split,)
        loader_train = data.DataLoader(train,**unpack)
        loader_valid = data.DataLoader(valid,**unpack)
        return loader_train,loader_valid
    elif train_valid_split == None:
        loader_test  = data.DataLoader(generator,**unpack)
        return loader_test,None
    else:
        raise NotImplementedError

def dataloader(dataset_name:str                     = 'CIFAR10',
               root:str                             = '../data',
               train:bool                           = True,
               transform:Optional[Callable]         = None,
               target_transform:Optional[Callable]  = None,
               download:bool                        = False,
               train_valid_split:Optional[List]     = None,
               batch_size:int                       = 8,
               num_workers:int                      = 2,
               shuffle:bool                         = True,
               return_path:bool                     = False,
               ) -> Tuple:
    """
    Download the datasets from PyTorch torchvision.datasets
    If no dataset file exists, specific "download=True"
    
    Parameters
    ---
    dataset_name:str, if not provided, we use images from a local directory
    root: the directory of the local images or the directory of the downloaded benckmark datasets
    train:bool, If True, creates dataset from training set, otherwise creates from test set
    trasnform:transforms, 
    target_transform:transforms
    download:bool,If true, downloads the dataset from the internet and puts it in root directory. If dataset is already downloaded, it is not downloaded again.
    train_valid_split:List[int], the number of training and validation examples for splitting
    batch_size:int, batch size for loading the data
    num_works:in, n_jobs
    shuffle:bool, shuffle the loading
    return_path:bool, not used
    
    Returns
    ---
    Tuple of dataloaders or dataloader + None
    """
    unpack1 = dict(root             = root,
                   transform        = transform,
                   target_transform = target_transform,
                   download         = download,
                   train            = train
                   )
    unpack2 = dict(batch_size       = batch_size,
                   num_workers      = num_workers,
                   shuffle          = shuffle,
                   )
    if dataset_name == 'CIFAR100': # subset of CIFAR10
        generator   = datasets.CIFAR10(**unpack1)
        loaders     = standard_dataset(generator, 
                                       train_valid_split,
                                       unpack2)
    elif dataset_name == 'CIFAR10':
        generator   = datasets.CIFAR100(**unpack1)
        loaders     = standard_dataset(generator, 
                                       train_valid_split,
                                       unpack2)
    elif dataset_name == None:
        loader = data_loader(data_root      = root,
                             augmentations  = transform,
                             return_path    = False,
                             **unpack2,)
        loaders = loader,None
    else:
        raise NotImplementedError
    return loaders

class customizedDataset(ImageFolder):
    """
    
    """
    def __getitem__(self, idx):
        original_tuple  = super(customizedDataset,self).__getitem__(idx)
        path            = self.imgs[idx][0]
        tuple_with_path = (original_tuple +  (path,))
        return tuple_with_path

def data_loader(data_root:str,
                augmentations:transforms    = None,
                batch_size:int              = 8,
                num_workers:int             = 1,
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
    Parameters
    ---
    data_root: str, the main folder
    augmentations: torchvision.transformers.Compose, steps of augmentation
    batch_size: int, batch size
    num_workers: int, CPU --> GPU carrier, number of CPUs
    shuffle: Boolean, whether to shuffle the order
    return_pth: Boolean, lod the image paths

    Returns
    ---
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

def optimizer_and_scheduler(params:List,
                            learning_rate:float     = 1e-4,
                            l2_regularization:float = 1e-16,
                            mode:str                = 'min',
                            factor:float            = .5,
                            patience:int            = 5,
                            threshold:float         = 1e-4,
                            min_lr:float            = 1e-8,
                            ) -> Union:
    """
    Build optimizer and scheduler

    Parameters
    ----------
    params : List of pytorch parameters
        The parameters we will modify during training.
    learning_rate : float, optional
        DESCRIPTION. The default is 1e-4.
    l2_regularization : float, optional
        DESCRIPTION. The default is 1e-16.
    mode : str, optional
        DESCRIPTION. The default is 'min'.
    factor : float, optional
        Learning rate decay factor. The default is .5.
    patience : int, optional
        DESCRIPTION. The default is 5.
    threshold : float, optional
        Tolerance of change during validation. The default is 1e-4.
    min_lr : float, optional
        DESCRIPTION. The default is 1e-8.

    Returns
    -------
    optimizer : torch.optim
    scheduler : torch.optim.lr_scheduler
        DESCRIPTION.

    """
    optimizer       = optim.Adam(params,
                                 lr             = learning_rate,
                                 weight_decay   = l2_regularization,
                                 )
    scheduler       = optim.lr_scheduler.ReduceLROnPlateau(optimizer    = optimizer,
                                                           mode         = mode,
                                                           factor       = factor,
                                                           patience     = patience,
                                                           threshold    = threshold,
                                                           min_lr       = min_lr,
                                                           verbose      = 1,
                                                           )
    return optimizer,scheduler

#candidate models
def candidates(model_name:str,pretrained:bool = True,) -> nn.Module:
    """
    A simple loader for the CNN backbone models

    Parameters
    ----------
    model_name : str
        DESCRIPTION.
    pretrained : bool, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    nn.Module
        A pretrained CNN model.

    """
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
            mobilenet_v3_l  = Tmodels.mobilenet_v3_large(pretrained = pretrained,
                                                         progress   = False,),
            # resnext50_32x4d = Tmodels.resnext50_32x4d(pretrained    = pretrained,
            #                                          progress       = False,),
            resnet50        = Tmodels.resnet50(pretrained           = pretrained,
                                              progress              = False,),
            )
    return picked_models[model_name]

def define_type(model_name:str) -> str:
    """
    We define the type of the pretrained CNN models for easier transfer learning

    Parameters
    ----------
    model_name : str
        DESCRIPTION.

    Returns
    -------
    str
        DESCRIPTION.

    """
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

def hidden_activation_functions(activation_func_name:str,num_parameters:int=3) -> nn.Module:
    """
    A simple loader for some of the nonlinear activation functions
    Parameters

    Parameters
    ----------
    activation_func_name : str
        DESCRIPTION.
    num_parameters : int
        I don't know how to use this yet.

    Returns
    -------
    nn.Module
        The activation function.

    """
    funcs = dict(relu       = nn.ReLU(),
                 selu       = nn.SELU(),
                 elu        = nn.ELU(),
                 celu       = nn.CELU(),
                 gelu       = nn.GELU(),
                 silu       = nn.SiLU(),
                 sigmoid    = nn.Sigmoid(),
                 tanh       = nn.Tanh(),
                 linear     = None,
                 leaky_relu = nn.LeakyReLU(),
                 hardshrink = nn.Hardshrink(lambd = .1),
                 softshrink = nn.Softshrink(lambd = .1),
                 tanhshrink = nn.Tanhshrink(),
                 # weight decay should not be used when learning aa for good performance.
                 prelu      = nn.PReLU(num_parameters=num_parameters,),
                 )
    return funcs[activation_func_name]

def noise_fuc(x:Tensor,noise_level:float = 1,scale:Tuple = (0,1)) -> Tensor:
    """
    add guassian noise to the images during agumentation procedures

    Parameters
    ----------
    x : Tensor
        Input images.
    noise_level : float, optional
        Standard deviation of the noise distribution. The default is 1.
    scale : Tuple, optional
        The range we scale the data to. 

    Returns
    -------
    x: Tensor
        DESCRIPTION.

    """
    if noise_level > 0:
        # generator = torch.distributions.normal.Normal(0,noise_level)
        generator = torch.distributions.half_normal.HalfNormal(scale = noise_level,)
        x = x + generator.sample(x.shape)
        # rescale x back to [0,1]
        x = (x - x.min()) / (x.max() - x.min())
        x = x * (scale[1] - scale[0]) + scale[0]
    return x

def simple_augmentations(image_resize:int   = 128,
                         noise_level        = None,
                         rotation:bool      = True,
                         gitter_color:bool  = False,
                         ):
    """
    Simple augmentation steps
    
    Parameters 
    ---
    image_resize: int, the height and width of the images
    noise_level: float or None, standard deviation of the Gaussian distribution the noise is sampled from
    rotation: bool, one of the augmentation methods, for object recognition only
    gitter_color: bool, one of the augmentation methods, for Gabor only
    
    Returns
    ---
    torchvision.transformer object
    """
    steps = [transforms.Resize((image_resize,image_resize)),
             transforms.Grayscale(num_output_channels=3)]
    
    if rotation and not gitter_color:
        steps.append(transforms.RandomHorizontalFlip(p = 0.5))
        steps.append(transforms.RandomRotation(45,))
        steps.append(transforms.RandomVerticalFlip(p = 0.5))
        steps.append(transforms.RandomPerspective(p = 0.5))
        steps.append(transforms.RandomAutocontrast(p = 0.5))
        # steps.append(transforms.RandomInvert(p = 0.5))
    elif gitter_color and not rotation:
        steps.append(transforms.RandomCrop((image_resize,image_resize)))
        steps.append(transforms.ColorJitter(brightness = 0.25,
                                            contrast = 0.25,
                                            saturation = 0.25,
                                            hue = 0.25,))
    # this step scale images to [0,1]
    steps.append(transforms.ToTensor())
    steps.append(transforms.Lambda(lambda x:noise_fuc(x,noise_level)))
    steps.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    # mean=[0., 0., 0.], std=[1., 1., 1.]
    # mean=[.5,.,5.,5], std=[.225,.225,.225]
    # mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    transform_steps = transforms.Compose(steps)
    return transform_steps

def determine_training_stops(net,
                             idx_epoch:int,
                             warmup_epochs:int,
                             valid_loss:Tensor,
                             counts: int        = 0,
                             device             = 'cpu',
                             best_valid_loss    = np.inf,
                             tol:float          = 1e-4,
                             f_name:str         = 'temp.h5',
                             ) -> Tuple[Tensor,int]:
    """
    A function in validation determining whether to stop training
    It only works after the warmup 

    Parameters
    ----------
    net : nn.Module
        DESCRIPTION.
    idx_epoch : int
        DESCRIPTION.
    warmup_epochs : int
        DESCRIPTION.
    valid_loss : Tensor
        DESCRIPTION.
    counts : int, optional
        DESCRIPTION. The default is 0.
    device : TYPE, optional
        DESCRIPTION. The default is 'cpu'.
    best_valid_loss : TYPE, optional
        DESCRIPTION. The default is np.inf.
    tol : float, optional
        DESCRIPTION. The default is 1e-4.
    f_name : str, optional
        DESCRIPTION. The default is 'temp.h5'.

    Returns
    -------
    best_valid_loss: Tensor
        DESCRIPTION.
    counts:int
        used for determine when to stop training
    """
    if idx_epoch > warmup_epochs: # warming up
        temp = valid_loss.cpu().clone().detach().type(torch.float64)
        if np.logical_and(temp < best_valid_loss,np.abs(best_valid_loss - temp) >= tol):
            best_valid_loss = valid_loss.cpu().clone().detach().type(torch.float64)
            torch.save(net.state_dict(),f_name)# why do i need state_dict()?
            counts = 0
        else:
            counts += 1
    return best_valid_loss,counts

def compute_image_loss(image_loss_func:Callable,
                       image_category:Tensor,
                       labels:Tensor,
                       device:str,
                       n_noise:int      = 0,
                       num_classes:int  = 2,
                       ) -> Tensor:
    """
    Compute the loss of predicting the image categories

    Parameters
    ----------
    image_loss_func : Callable
        DESCRIPTION.
    image_category : Tensor
        DESCRIPTION.
    labels : Tensor
        DESCRIPTION.
    device : str
        DESCRIPTION.
    n_noise : int, optional
        DESCRIPTION. The default is 0.
    num_classes : int, optional
        DESCRIPTION. The default is 10.

    Returns
    -------
    image_loss: Tensor
        DESCRIPTION.

    """
    if "Binary Cross Entropy" in image_loss_func.__doc__:
        labels = F.one_hot(labels,num_classes = num_classes,)
        labels = labels.float()
        if n_noise > 0:
            noisy_labels    = torch.ones(labels.shape) * (1/num_classes)
            noisy_labels    = noisy_labels[:n_noise]
            labels          = torch.cat([labels.to(device),noisy_labels.to(device)])
        # print(image_category.shape,labels.shape)
        image_loss = image_loss_func(image_category.to(device),
                                     labels.view(image_category.shape).to(device)
                                     )
    elif "negative log likelihood loss" in image_loss_func.__doc__:
        labels = labels.long()
        image_loss = image_loss_func(torch.log(image_category).to(device),
                                     labels.to(device))
    return image_loss

def compute_kl_divergence(mu:Tensor,log_var:Tensor,) -> Tensor:
        """
        Q(z|X) has mean of `mu` and std of `exp(log_var)`
        kld = E[log(Q(z|X)) - log(P(z))], where P() is the function P(z|x)
        https://towardsdatascience.com/variational-autoencoder-demystified-with-pytorch-implementation-3a06bee395ed
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        Inputs
        ---
        
        mu:torch.tensor
        log_var:torch.tensor
        
        Outputs
        ---
        KLD_loss:torch.tensor
        """
        KLD_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        return KLD_loss
    
def compute_reconstruction_loss(hidden_representation:Tensor,
                                reconstruct:Tensor,
                                loss_func:nn.Module,
                                n_noise:int = 0,
                                device = 'cpu',
                                ) -> Tensor:
    """
    

    Parameters
    ----------
    hidden_representation : Tensor
        DESCRIPTION.
    reconstruct : Tensor
        DESCRIPTION.
    loss_func : nn.Module
        DESCRIPTION.
    n_noise : int, optional
        DESCRIPTION. The default is 0.
    device : TYPE, optional
        DESCRIPTION. The default is 'cpu'.

    Returns
    -------
    Loss : Tensor
        DESCRIPTION.

    """
    if "similar or dissimilar" in loss_func.__doc__:
        """
        So the reconstructed tensor should be similar to the hidden representation
        but disimilar from a random tensor
        """
        a = hidden_representation - hidden_representation.mean(1).view(-1,1)
        b = reconstruct - reconstruct.mean(1).view(-1,1)
        if n_noise > 0:
            # we label the last n_noise hidden representations as "-1" to tell
            # the loss function these are random embeddings,the reconstruction
            # should be disimilar to them
            y_ones = torch.ones(hidden_representation.shape[0])
            y_ones[-n_noise:] = -1
            return loss_func(a.to(device),
                             b.to(device),y_ones.to(device))
        else:
            concat_recon = torch.cat([reconstruct.to(device),reconstruct.to(device)])
            dist_noise = torch.distributions.normal.Normal(hidden_representation.mean(),
                                                           hidden_representation.std(),)
            concat_hidden = torch.cat([hidden_representation.to(device),
                                       dist_noise.sample(hidden_representation.shape).to(device)])
            y_ones = torch.ones(hidden_representation.shape[0])
            y = torch.cat([y_ones,
                           torch.ones(hidden_representation.shape[0]) * -1])
            return loss_func(concat_recon.to(device),concat_hidden.to(device),y.to(device))
    else:
        """
        Or else, we just want to minimize the distance between the two embeddings
        """
        return loss_func(reconstruct.to(device),hidden_representation.to(device))

def clf_vae_train_loop(net:nn.Module,
                       dataloader:data.DataLoader,
                       optimizers:List,
                       image_loss_func:nn.Module    = nn.NLLLoss(),
                       recon_loss_func:nn.Module    = nn.MSELoss(),
                       transform:transforms         = None,
                       image_resize:int             = 128,
                       n_noise:int                  = 0,
                       device                       = 'cpu',
                       idx_epoch:int                = 0,
                       print_train:bool             = True,
                       beta:float                   = 1.,
                       ):
    """
    

    Parameters
    ----------
    net : nn.Module
        DESCRIPTION.
    dataloader : data.DataLoader
        DESCRIPTION.
    optimizers : List
        DESCRIPTION.
    image_loss_func : nn.Module, optional
        DESCRIPTION. The default is nn.NLLLoss().
    recon_loss_func : nn.Module, optional
        DESCRIPTION. The default is nn.MSELoss().
    transform : transforms, optional
        DESCRIPTION. The default is None.
    image_resize : int, optional
        DESCRIPTION. The default is 128.
    n_noise : int, optional
        DESCRIPTION. The default is 0.
    device : TYPE, optional
        DESCRIPTION. The default is 'cpu'.
    idx_epoch : int, optional
        DESCRIPTION. The default is 0.
    print_train : bool, optional
        DESCRIPTION. The default is True.
    beta : float, optional
        DESCRIPTION. The default is 1..

    Returns
    -------
    net : TYPE
        DESCRIPTION.
    train_loss : TYPE
        DESCRIPTION.

    """
    optimizer1,optimizer2 = optimizers
    net.train(True)
    train_loss  = 0.
    iterator    = tqdm(enumerate(dataloader))
    for ii,(batch_features,batch_labels) in iterator:
        if n_noise > 0:
            # in order to have desired classification behavior, which is to predict
            # chance when no signal is present, we manually add some noise samples
            noise_generator = datasets.FakeData(size = n_noise,
                                                image_size = (3,image_resize,image_resize),
                                                num_classes = 1,
                                                transform = transform,
                                                # random_offset = 12345,
                                                )
            noisy_features  = torch.cat([item[0][None,:] for item in noise_generator])
            
            batch_features  = torch.cat([batch_features,noisy_features])
        # zero grad
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        # forward pass
        (reconstruction,
         extracted_features,
         z,mu,log_var,
         hidden_representation,
         image_category)  = net(batch_features.to(device))
        # compute loss
        ## image classification loss
        image_loss      = compute_image_loss(
                                        image_loss_func = image_loss_func,
                                        image_category  = image_category.to(device),
                                        labels          = batch_labels.to(device),
                                        device          = device,
                                        n_noise         = n_noise,
                                        )
        ## reconstruction loss
        recon_loss      = compute_reconstruction_loss(
                                                    hidden_representation,
                                                    reconstruction,
                                                    recon_loss_func,
                                                    n_noise = n_noise,
                                                    device = device,
                                                    )
        ## KLD loss
        kld_loss        = compute_kl_divergence(mu, log_var)
        
        # backpropagation
        loss_batch      = image_loss + recon_loss + beta * kld_loss
        loss_batch.backward()
        # modify the weights
        optimizer1.step()
        optimizer2.step()
        # record the loss of a mini-batch
        train_loss += loss_batch
        
        if print_train:
            iterator.set_description(f'epoch {idx_epoch+1:3.0f}-{ii + 1:4.0f}/{100*(ii+1)/len(dataloader):2.3f}%,train loss = {train_loss/(ii+1):2.6f}')
    
    return net,train_loss

def clf_vae_valid_loop(net:nn.Module,
                       dataloader:data.DataLoader,
                       image_loss_func:nn.Module    = nn.NLLLoss(),
                       recon_loss_func:nn.Module    = nn.MSELoss(),
                       device                       = 'cpu',
                       idx_epoch:int                = 0,
                       print_train:bool             = True,
                       beta:float                   = 1.,
                       ):
    """
    

    Parameters
    ----------
    net : nn.Module
        DESCRIPTION.
    dataloader : data.DataLoader
        DESCRIPTION.
    image_loss_func : nn.Module, optional
        DESCRIPTION. The default is nn.NLLLoss().
    recon_loss_func : nn.Module, optional
        DESCRIPTION. The default is nn.MSELoss().
    device : TYPE, optional
        DESCRIPTION. The default is 'cpu'.
    idx_epoch : int, optional
        DESCRIPTION. The default is 0.
    print_train : bool, optional
        DESCRIPTION. The default is True.
    beta : float, optional
        DESCRIPTION. The default is 1..

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    net.eval()
    valid_loss   = 0.
    image_losses = 0.
    recon_losses = 0.
    kld_losses   = 0.
    iterator    = tqdm(enumerate(dataloader))
    y_true      = []
    y_pred      = []
    with torch.no_grad():
        for ii,(batch_features,batch_labels) in iterator:
            # forward pass
            (reconstruction,
             extracted_features,
             z,mu,log_var,
             hidden_representation,
             image_category)  = net(batch_features.to(device))
            y_true.append(batch_labels)
            y_pred.append(image_category)
            # compute loss
            ## image classification loss
            image_loss      = compute_image_loss(
                                            image_loss_func = image_loss_func,
                                            image_category  = image_category.to(device),
                                            labels          = batch_labels.to(device),
                                            device          = device,
                                            )
            image_losses += image_loss
            ## reconstruction loss
            recon_loss      = compute_reconstruction_loss(
                                                        hidden_representation,
                                                        reconstruction,
                                                        recon_loss_func,
                                                        device = device,
                                                        )
            recon_losses += recon_loss
            ## KLD loss
            kld_loss        = compute_kl_divergence(mu, log_var)
            kld_losses += beta * kld_loss
            # losses
            loss_batch      = image_loss + recon_loss + beta * kld_loss
            valid_loss += loss_batch
            if print_train:
                iterator.set_description(f'epoch {idx_epoch+1:3.0f}-{ii + 1:4.0f}/{100*(ii+1)/len(dataloader):2.3f}%,valid loss = {valid_loss/(ii+1):2.6f}')
    return (valid_loss/len(dataloader),
            torch.cat(y_true),
            torch.cat(y_pred),
            image_losses/len(dataloader),
            recon_losses/len(dataloader),
            kld_losses/len(dataloader))
def clf_vae_train_valid(net:nn.Module,
                        dataloader_train:data.DataLoader,
                        dataloader_valid:data.DataLoader,
                        optimizers:List,
                        schedulers:List,
                        transform:transforms                = None,
                        image_loss_func:nn.Module           = nn.NLLLoss(),
                        recon_loss_func:nn.Module           = nn.MSELoss(),
                        image_resize                        = 128,
                        n_epochs:int                        = int(1e3),
                        device                              = 'cpu',
                        print_train:bool                    = True,
                        warmup_epochs:int                   = 10,
                        tol:float                           = 1e-4,
                        f_name:str                          = 'temp.h5',
                        patience:int                        = 10,
                        n_noise:int                         = 0,
                        beta:float                          = 1.,
                        ):
    """
    

    Parameters
    ----------
    net : nn.Module
        DESCRIPTION.
    dataloader_train : data.DataLoader
        DESCRIPTION.
    dataloader_valid : data.DataLoader
        DESCRIPTION.
    optimizers : List of optimizers
        DESCRIPTION.
    schedulers : List of schedulers
        DESCRIPTION.
    transform : transforms, optional
        DESCRIPTION. The default is None.
    image_loss_func : nn.Module, optional
        DESCRIPTION. The default is nn.NLLLoss().
    recon_loss_func : nn.Module, optional
        DESCRIPTION. The default is nn.MSELoss().
    image_resize : TYPE, optional
        DESCRIPTION. The default is 128.
    n_epochs : int, optional
        DESCRIPTION. The default is int(1e3).
    device : TYPE, optional
        DESCRIPTION. The default is 'cpu'.
    print_train : bool, optional
        DESCRIPTION. The default is True.
    warmup_epochs : int, optional
        DESCRIPTION. The default is 10.
    tol : float, optional
        DESCRIPTION. The default is 1e-4.
    f_name : str, optional
        DESCRIPTION. The default is 'temp.h5'.
    patience : int, optional
        DESCRIPTION. The default is 10.
    n_noise : int, optional
        DESCRIPTION. The default is 0.
    beta : float, optional
        DESCRIPTION. The default is 1..

    Returns
    -------
    net : nn.Module
        Trained model
    losses : List of Tensors
        losses

    """
    torch.random.manual_seed(12345)
    scheduler1,scheduler2   = schedulers
    best_valid_loss         = np.inf
    losses                  = []
    counts                  = 0
    for idx_epoch in range(n_epochs):
        net,train_loss      = clf_vae_train_loop(
                                net             = net,
                                dataloader      = dataloader_train,
                                optimizers      = optimizers,
                                image_loss_func = image_loss_func,
                                recon_loss_func = recon_loss_func,
                                n_noise         = n_noise,
                                transform       = transform,
                                image_resize    = image_resize,
                                device          = device,
                                idx_epoch       = idx_epoch,
                                print_train     = print_train,
                                beta            = beta,
                                )
        (valid_loss,
         y_true,y_pred,
         image_loss,recon_loss,kld_loss) = clf_vae_valid_loop(
                                net             = net,
                                dataloader      = dataloader_valid,
                                image_loss_func = image_loss_func,
                                recon_loss_func = recon_loss_func,
                                device          = device,
                                idx_epoch       = idx_epoch,
                                print_train     = print_train,
                                beta            = beta,
                                )
        if idx_epoch > warmup_epochs:
            scheduler1.step(image_loss)
            scheduler2.step(recon_loss + kld_loss)
        best_valid_loss,counts = determine_training_stops(net,
                                                          idx_epoch,
                                                          warmup_epochs,
                                                          valid_loss,
                                                          counts            = counts,
                                                          device            = device,
                                                          best_valid_loss   = best_valid_loss,
                                                          tol               = tol,
                                                          f_name            = f_name,
                                                          )
        # calculate accuracy
        try:
            accuracy = roc_auc_score(y_true.detach().cpu().numpy(),
                                     y_pred.detach().cpu().numpy()
                                     )
        except:
            accuracy = torch.sum(y_true.to(device) == y_pred.max(1)[1].to(device)) / y_true.shape[0]
        print(f'''
epoch {idx_epoch+1:3.0f} 
          validation accuracy = {accuracy:2.4f},
          image loss          = {image_loss.detach().cpu().numpy():.4f},
          reconstruction loss = {recon_loss.detach().cpu().numpy():.4f},
          VAE loss            = {kld_loss.detach().cpu().numpy():.4f},
          counts              = {counts}''')
        if counts >= patience:#(len(losses) > patience) and (len(set(losses[-patience:])) == 1):
            break
    losses.append(best_valid_loss.detach().cpu().numpy())
    return net,losses

if __name__ == "__main__":
    pass