#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 13:03:04 2022

@author: nmei

I didn't know spyder has a quick documentation feature

"""
from typing import List, Callable, Union, Any, TypeVar, Tuple, List, Optional
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
from torch          import nn,no_grad
from torch.utils    import data
from torch.nn       import functional as F
from torch          import optim
from torch.autograd import Variable

from torchvision.datasets import ImageFolder
from torchvision          import transforms,datasets
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

def standard_dataset(generator:datasets,
                     train_valid_split:List,
                     unpack:dict,
                     ) -> Tuple:
    """
    
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
    unpack2 = dict(batch_size   = batch_size,
                   num_workers  = num_workers,
                   shuffle      = shuffle,
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
        loader = data_loader(data_root = root,
                             augmentations = transform,
                             return_path = False,
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
        path = self.imgs[idx][0]
        tuple_with_path = (original_tuple +  (path,))
        return tuple_with_path

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

#candidate models
def candidates(model_name:str,pretrained:bool = True,) -> nn.Module:
    """
    A simple loader for the CNN backbone models
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
            # resnext50_32x4d = Tmodels.resnext50_32x4d(pretrained    = pretrained,
            #                                          progress       = False,),
            resnet50        = Tmodels.resnet50(pretrained           = pretrained,
                                              progress              = False,),
            )
    return picked_models[model_name]

def define_type(model_name:str) -> str:
    """
    

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

def hidden_activation_functions(activation_func_name:str) -> nn.Module:
    """
    A simple loader for some of the nonlinear activation functions
    Parameters
    ----------
    activation_func_name : str

    Returns
    -------
    nn.Module
        an activation function

    """
    funcs = dict(relu       = nn.ReLU(),
                 selu       = nn.SELU(),
                 elu        = nn.ELU(),
                 sigmoid    = nn.Sigmoid(),
                 tanh       = nn.Tanh(),
                 linear     = None,
                 leaky_relu = nn.LeakyReLU(),
                 )
    return funcs[activation_func_name]

def noise_fuc(x,noise_level = 1):
    """
    add guassian noise to the images during agumentation procedures

    Parameters
    --------------------
    x: torch.tensor, batch_size x 3 x height x width
    noise_level: float, standard deviation of the gaussian distribution
    """
    generator = torch.distributions.normal.Normal(0,noise_level)
    return x + generator.sample(x.shape)

def simple_augmentations(image_resize   = 128,
                         noise_level    = None,
                         rotation       = True,
                         gitter_color   = False,
                         ):
    """
    Simple augmentation steps
    
    Parameters 
    ---
    image_resize: int, the height and width of the images
    noise_level: float, standard deviation of the Gaussian distribution the noise is sampled from
    rotation: bool, one of the augmentation methods, for object recognition only
    gitter_color: bool, one of the augmentation methods, for Gabor only
    
    Returns
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

def vae_train_loop(net,
                   dataloader,
                   optimizer,
                   recon_loss_func:Optional[nn.Module]  = None,
                   idx_epoch:int                        = 0,
                   device                               = 'cpu',
                   print_train:bool                     = True,
                   ) -> Union[nn.Module,Tensor]:
    """
    Train a variational autoencoder
    
    Parameters
    ----------
    net : nn.Module
        DESCRIPTION.
    dataloader : torch.data.DataLoader
        DESCRIPTION.
    optimizer : optim
        DESCRIPTION.
    recon_loss_func : Optional[nn.Module], optional
        DESCRIPTION. The default is None.
    idx_epoch : int, optional
        DESCRIPTION. The default is 0.
    device : str or torch.device, optional
        DESCRIPTION. The default is 'cpu'.
    print_train : bool, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    net : nn.Module
        DESCRIPTION.
    train_loss : Tensor
        DESCRIPTION.

    """
    if recon_loss_func == None:
        recon_loss_func = nn.MSELoss()
    net.train(True)
    train_loss  = 0.
    iterator    = tqdm(enumerate(dataloader))
    for ii,(batch_features,batch_labels) in iterator:
        # zero grad
        optimizer.zero_grad()
        # forward pass
        (reconstruction,
         hidden_representation,
         z,mu,log_var)  = net(batch_features.to(device))
        # reconstruction loss
        recon_loss      = recon_loss_func(batch_features.to(device),reconstruction)
        # variational loss
        kld_loss        = net.kl_divergence(z, mu, log_var)
        loss_batch      = recon_loss + kld_loss
        # backpropagation
        loss_batch.backward()
        # modify the weights
        optimizer.step()
        # record the loss of a mini-batch
        train_loss += loss_batch.data
        if print_train:
            iterator.set_description(f'epoch {idx_epoch+1:3.0f}-{ii + 1:4.0f}/{100*(ii+1)/len(dataloader):2.3f}%,train loss = {train_loss/(ii+1):2.6f}')
    return net,train_loss

def vae_valid_loop(net,
                   dataloader,
                   recon_loss_func:Optional[nn.Module]  = None,
                   idx_epoch:int                        = 0,
                   device                               = 'cpu',
                   print_train:bool                     = True,
                   ) -> Tensor:
    """
    validation process of the model

    Parameters
    ----------
    net : nn.Module
        DESCRIPTION.
    dataloader : Callable
        DESCRIPTION.
    recon_loss_func : Optional[nn.Module], optional
        DESCRIPTION. The default is None.
    idx_epoch : int, optional
        DESCRIPTION. The default is 0.
    device : str or torch.device, optional
        DESCRIPTION. The default is 'cpu'.
    print_train : bool, optional
        DESCRIPTION. The default is True.
     

    Returns
    -------
    Tensor
        DESCRIPTION.

    """
    if recon_loss_func == None:
        recon_loss_func = nn.MSELoss()
    net.eval()
    valid_loss  = 0.
    iterator    = tqdm(enumerate(dataloader))
    with torch.no_grad():
        for ii,(batch_features,batch_labels) in iterator:
             (reconstruction,
              hidden_representation,
              z,mu,log_var) = net(batch_features.to(device))
             recon_loss     = recon_loss_func(batch_features.to(device),reconstruction)
             kld_loss       = net.kl_divergence(z, mu, log_var)
             loss_batch     = recon_loss + kld_loss
             valid_loss += loss_batch.data
             if print_train:
                 iterator.set_description(f'epoch {idx_epoch+1:3.0f}-{ii + 1:4.0f}/{100*(ii+1)/len(dataloader):2.3f}%,valid loss = {valid_loss/(ii+1):2.6f}')
    return valid_loss

def vae_train_valid(net,
                    dataloader_train,
                    dataloader_valid,
                    optimizer,
                    n_epochs:int                        = int(1e3),
                    recon_loss_func:Optional[nn.Module] = None,
                    device                              = 'cpu',
                    print_train:bool                    = True,
                    warmup_epochs:int                   = 10,
                    tol:float                           = 1e-4,
                    f_name:str                          = 'temp.h5',
                    patience:int                        = 10,
                    ) -> Union[nn.Module,Tensor]:
    """
    Train and validation process of the VAE

    Parameters
    ----------
    net : nn.Module
        DESCRIPTION.
    dataloader_train : callable
        DESCRIPTION.
    dataloader_valid : callable
        DESCRIPTION.
    optimizer : callable
        DESCRIPTION.
    n_epochs : int, optional
        DESCRIPTION. The default is int(1e3).
    recon_loss_func : Optional[nn.Module], optional
        DESCRIPTION. The default is None.
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

    Returns
    -------
    net : nn.Module
        DESCRIPTION.
    losses : List[Tensor]
        DESCRIPTION.

    """
    torch.random.manual_seed(12345)
    
    best_valid_loss     = np.inf
    losses              = []
    counts              = 0
    for idx_epoch in range(n_epochs):
        _ = vae_train_loop(net,
                           dataloader_train,
                           optimizer,
                           recon_loss_func  = recon_loss_func,
                           idx_epoch        = idx_epoch,
                           device           = device,
                           print_train      = print_train,
                           )
        valid_loss = vae_valid_loop(net,
                                    dataloader_valid,
                                    recon_loss_func = recon_loss_func,
                                    idx_epoch       = idx_epoch,
                                    device          = device,
                                    print_train     = print_train,
                                    )
        best_valid_loss,counts = determine_training_stops(net,
                                                          idx_epoch,
                                                          warmup_epochs,
                                                          valid_loss,
                                                          device            = device,
                                                          best_valid_loss   = best_valid_loss,
                                                          tol               = tol,
                                                          f_name            = f_name,
                                                          )
        if counts >= patience:#(len(losses) > patience) and (len(set(losses[-patience:])) == 1):
            break
    losses.append(best_valid_loss)
    return net,losses

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
                       num_classes:int  = 10,
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
            noisy_labels    = torch.ones(labels.shape) * 0.5
            noisy_labels    = noisy_labels[:n_noise]
            labels          = torch.cat([labels.to(device),noisy_labels.to(device)])
        print(image_category.shape,labels.shape)
        image_loss = image_loss_func(image_category.to(device),
                                     labels.view(image_category.shape).to(device)
                                     )
    elif "negative log likelihood loss" in image_loss_func.__doc__:
        labels = labels.long()
        image_loss = image_loss_func(torch.log(image_category).to(device),
                                     labels[:,-1].to(device))
    return image_loss


def clf_train_loop(net:nn.Module,
                   dataloader:data.DataLoader,
                   optimizer:Callable,
                   image_loss_func:Optional[nn.Module]  = None,
                   idx_epoch:int                        = 0,
                   device                               = 'cpu',
                   print_train:bool                     = True,
                   n_noise:int                          = 0,
                   ) -> Union[nn.Module,Tensor]:
    """
    

    Parameters
    ----------
    net : nn.Module
        DESCRIPTION.
    dataloader : data.DataLoader
        DESCRIPTION.
    optimizer : Callable
        DESCRIPTION.
    image_loss_func : Optional[nn.Module], optional
        DESCRIPTION. The default is None.
    idx_epoch : int, optional
        DESCRIPTION. The default is 0.
    device : TYPE, optional
        DESCRIPTION. The default is 'cpu'.
    print_train : bool, optional
        DESCRIPTION. The default is True.
    n_noise : int, optional
        DESCRIPTION. The default is 0.

    Returns
    -------
    net : nn.Module
        DESCRIPTION.
    train_loss : Tensor
        DESCRIPTION.

    """
    if image_loss_func == None:
        image_loss_func = nn.BCELoss()
    net.train(True)
    train_loss  = 0.
    iterator    = tqdm(enumerate(dataloader))
    for ii,(batch_features,batch_labels) in iterator:
        if n_noise > 0:
            # in order to have desired classification behavior, which is to predict
            # chance when no signal is present, we manually add some noise samples
            noise_generator = torch.distributions.normal.Normal(batch_features.mean(),
                                                                batch_features.std())
            noisy_features  = noise_generator.sample(batch_features.shape)[:n_noise]
            
            batch_features  = torch.cat([batch_features,noisy_features])
        # zero grad
        optimizer.zero_grad()
        # forward pass
        (reconstruction,
         image_category)  = net(batch_features.to(device))
        # compute loss
        loss_batch      = compute_image_loss(
                                        image_loss_func,
                                        image_category,
                                        batch_labels.to(device),
                                        device,
                                        n_noise,
                                        )
        # backpropagation
        loss_batch.backward()
        # modify the weights
        optimizer.step()
        # record the loss of a mini-batch
        train_loss += loss_batch.data
        if print_train:
            iterator.set_description(f'epoch {idx_epoch+1:3.0f}-{ii + 1:4.0f}/{100*(ii+1)/len(dataloader):2.3f}%,train loss = {train_loss/(ii+1):2.6f}')
    return net,train_loss

def clf_valid_loop(net:nn.Module,
                   dataloader:data.DataLoader,
                   image_loss_func:Optional[nn.Module]  = None,
                   idx_epoch:int                        = 0,
                   device                               = 'cpu',
                   print_train:bool                     = True,
                   ) -> Tensor:
    """
    

    Parameters
    ----------
    net : nn.Module
        DESCRIPTION.
    dataloader : data.DataLoader
        DESCRIPTION.
    image_loss_func : Optional[nn.Module], optional
        DESCRIPTION. The default is None.
    idx_epoch : int, optional
        DESCRIPTION. The default is 0.
    device : TYPE, optional
        DESCRIPTION. The default is 'cpu'.
    print_train : bool, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    valid_loss : Tensor
        DESCRIPTION.

    """
    if image_loss_func == None:
        image_loss_func = nn.BCELoss()
    net.eval()
    valid_loss  = 0.
    iterator    = tqdm(enumerate(dataloader))
    with torch.no_grad():
        for ii,(batch_features,batch_labels) in iterator:
             (reconstruction,
              image_category)  = net(batch_features.to(device))
             # compute loss
             loss_batch      = compute_image_loss(
                                            image_loss_func,
                                            image_category,
                                            batch_labels.to(device),
                                            device,
                                            )
             valid_loss += loss_batch
             if print_train:
                 iterator.set_description(f'epoch {idx_epoch+1:3.0f}-{ii + 1:4.0f}/{100*(ii+1)/len(dataloader):2.3f}%,valid loss = {valid_loss/(ii+1):2.6f}')
    return valid_loss

def clf_train_valid(net:nn.Module,
                    dataloader_train:data.DataLoader,
                    dataloader_valid:data.DataLoader,
                    optimizer:torch.optim,
                    n_epochs:int                        = int(1e3),
                    image_loss_func:Optional[nn.Module] = None,
                    device                              = 'cpu',
                    print_train:bool                    = True,
                    warmup_epochs:int                   = 10,
                    tol:float                           = 1e-4,
                    f_name:str                          = 'temp.h5',
                    patience:int                        = 10,
                    n_noise:int                         = 0,
                    ) -> Union[nn.Module,List]:
    """
    

    Parameters
    ----------
    net : nn.Module
        DESCRIPTION.
    dataloader_train : data.DataLoader
        DESCRIPTION.
    dataloader_valid : data.DataLoader
        DESCRIPTION.
    optimizer : torch.optim
        DESCRIPTION.
    n_epochs : int, optional
        DESCRIPTION. The default is int(1e3).
    image_loss_func : Optional[nn.Module], optional
        DESCRIPTION. The default is None.
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

    Returns
    -------
    net : nn.Module
        DESCRIPTION.
    losses : List of Tensor
        DESCRIPTION.

    """
    torch.random.manual_seed(12345)
    
    best_valid_loss     = np.inf
    losses              = []
    counts              = 0
    for idx_epoch in range(n_epochs):
        _ = clf_train_loop(net,
                           dataloader_train,
                           optimizer,
                           image_loss_func  = image_loss_func,
                           idx_epoch        = idx_epoch,
                           device           = device,
                           print_train      = print_train,
                           n_noise          = n_noise,
                           )
        valid_loss = clf_valid_loop(net,
                                    dataloader_valid,
                                    image_loss_func = image_loss_func,
                                    idx_epoch       = idx_epoch,
                                    device          = device,
                                    print_train     = print_train,
                                    )
        best_valid_loss,counts = determine_training_stops(net,
                                                          idx_epoch,
                                                          warmup_epochs,
                                                          valid_loss,
                                                          device            = device,
                                                          best_valid_loss   = best_valid_loss,
                                                          tol               = tol,
                                                          f_name            = f_name,
                                                          )
        if counts >= patience:#(len(losses) > patience) and (len(set(losses[-patience:])) == 1):
            break
    losses.append(best_valid_loss)
    return net,losses

if __name__ == "__main__":
    pass