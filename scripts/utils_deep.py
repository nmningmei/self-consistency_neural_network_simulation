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
    Generate optimizer and scheduler

    Parameters
    ----------
    params : List
        DESCRIPTION.
    learning_rate : float, optional
        DESCRIPTION. The default is 1e-4.
    l2_regularization : float, optional
        DESCRIPTION. The default is 1e-16.
    mode : str, optional
        DESCRIPTION. The default is 'min'.
    factor : float, optional
        DESCRIPTION. The default is .5.
    patience : int, optional
        DESCRIPTION. The default is 5.
    threshold : float, optional
        DESCRIPTION. The default is 1e-4.
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
                 prelu      = nn.PReLU(num_parameters=3,),
                 )
    return funcs[activation_func_name]

def noise_fuc(x:Tensor,noise_level:float = 1,scale:Tuple = (0,1)) -> Tensor:
    """
    add guassian noise to the images during agumentation procedures

    Parameters
    ----------
    x : Tensor
        DESCRIPTION.
    noise_level : float, optional
        DESCRIPTION. The default is 1.
    scale : Tuple, optional
        DESCRIPTION. 

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
    
def compute_reconstruction_loss(x:Tensor,reconstruct:Tensor,loss_func:nn.Module,) -> Tensor:
    return loss_func(x,reconstruct)

def vae_train_loop(net,
                   dataloader,
                   optimizer,
                   recon_loss_func:Optional[nn.Module]  = None,
                   idx_epoch:int                        = 0,
                   device                               = 'cpu',
                   print_train:bool                     = True,
                   beta:float                           = 1.,
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
    beta: float
        weight of variational loss
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
        loss_batch      = recon_loss + beta * kld_loss
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
                   classifier : Optional[nn.Module]     = None,
                   ) -> Tuple:
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
    classifier : Optional[nn.Module], optional
        It is used for a metric of the VAE. The default is None.

    Returns
    -------
    valid_loss:Tensor
        DESCRIPTION.
    y_true: Optional[Tensor]
    y_pred: Optional[Tensor]
    """
    if recon_loss_func == None:
        recon_loss_func = nn.MSELoss()
    net.eval()
    valid_loss  = 0.
    iterator    = tqdm(enumerate(dataloader))
    y_true      = []
    y_pred      = []
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
             if classifier is not None:
                 _,image_category = classifier(reconstruction.to(device))
                 y_true.append(batch_labels)
                 y_pred.append(image_category)
    if classifier is not None:
        y_true = torch.cat(y_true)
        y_pred = torch.cat(y_pred)
    return valid_loss,y_true,y_pred

def vae_train_valid(net,
                    dataloader_train,
                    dataloader_valid,
                    optimizer,
                    scheduler,
                    n_epochs:int                        = int(1e3),
                    recon_loss_func:Optional[nn.Module] = None,
                    device                              = 'cpu',
                    print_train:bool                    = True,
                    warmup_epochs:int                   = 10,
                    tol:float                           = 1e-4,
                    f_name:str                          = 'temp.h5',
                    patience:int                        = 10,
                    classifier : Optional[nn.Module]    = None,
                    beta                                = 1.,
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
    scheduler : torch.optim
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
    classifier : Optional[nn.Module], optional
        It is used for a metric of the VAE. The default is None.
    beta : float
        weight of variational loss
    
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
    # adjust_lr           = True
    for idx_epoch in range(n_epochs):
        _ = vae_train_loop(net,
                           dataloader_train,
                           optimizer,
                           recon_loss_func  = recon_loss_func,
                           idx_epoch        = idx_epoch,
                           device           = device,
                           print_train      = print_train,
                           beta             = beta,
                           )
        valid_loss,y_true,y_pred = vae_valid_loop(net,
                                    dataloader_valid,
                                    recon_loss_func = recon_loss_func,
                                    idx_epoch       = idx_epoch,
                                    device          = device,
                                    print_train     = print_train,
                                    classifier      = classifier,
                                    )
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
        scheduler.step(valid_loss)
        if classifier is not None:
            accuracy = torch.sum(y_true.to(device) == y_pred.max(1)[1].to(device)) / y_true.shape[0]
            if print_train:
                print(f'\nepoch {idx_epoch+1:3.0f} validation accuracy = {accuracy:2.4f},counts = {counts}')
        # if idx_epoch + 1 > warmup_epochs and adjust_lr:
        #     optimizer.param_groups[0]['lr'] /= 10
        #     adjust_lr = False
        if counts >= patience:#(len(losses) > patience) and (len(set(losses[-patience:])) == 1):
            break
    losses.append(best_valid_loss)
    return net,losses

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
    Valid_loss:Tensor
        DESCRIPTION.
    y_true: Tensor, (n_samples,)
    y_pred: Tensor, (n_samples,n_classes)
    """
    if image_loss_func == None:
        image_loss_func = nn.BCELoss()
    net.eval()
    valid_loss  = 0.
    iterator    = tqdm(enumerate(dataloader))
    y_true      = []
    y_pred      = []
    with torch.no_grad():
        for ii,(batch_features,batch_labels) in iterator:
             (reconstruction,
              image_category)  = net(batch_features.to(device))
             y_true.append(batch_labels)
             y_pred.append(image_category)
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
    return valid_loss,torch.cat(y_true),torch.cat(y_pred)

def clf_train_valid(net:nn.Module,
                    dataloader_train:data.DataLoader,
                    dataloader_valid:data.DataLoader,
                    optimizer:torch.optim,
                    scheduler:torch.optim,
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
    scheduler : torch.optim
        learning rate scheduler
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
    # adjust_lr           = True
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
        valid_loss,y_true,y_pred = clf_valid_loop(net,
                                    dataloader_valid,
                                    image_loss_func = image_loss_func,
                                    idx_epoch       = idx_epoch,
                                    device          = device,
                                    print_train     = print_train,
                                    )
        scheduler.step(valid_loss)
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
        # if idx_epoch + 1 > warmup_epochs and adjust_lr:
        #     optimizer.param_groups[0]['lr'] /= 10
        #     adjust_lr = False
        
        # calculate accuracy
        accuracy = torch.sum(y_true.to(device) == y_pred.max(1)[1].to(device)) / y_true.shape[0]
        print(f'\nepoch {idx_epoch+1:3.0f} validation accuracy = {accuracy:2.4f},counts = {counts}')
        if counts >= patience:#(len(losses) > patience) and (len(set(losses[-patience:])) == 1):
            break
    losses.append(best_valid_loss)
    return net,losses

def clf_vae_train_loop(net:nn.Module,
                       dataloader:data.DataLoader,
                       optimizers:List,
                       image_loss_func:nn.Module = nn.NLLLoss(),
                       recon_loss_func:nn.Module = nn.MSELoss(),
                       n_noise:int = 0,
                       device = 'cpu',
                       idx_epoch:int = 0,
                       print_train:bool = True,
                       beta:float = 1.,
                       ):
    optimizer1,optimizer2 = optimizers
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
                                                    batch_features.to(device),
                                                    reconstruction.to(device),
                                                    recon_loss_func,
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
                       image_loss_func:nn.Module = nn.NLLLoss(),
                       recon_loss_func:nn.Module = nn.MSELoss(),
                       device = 'cpu',
                       idx_epoch:int = 0,
                       print_train:bool = True,
                       beta:float = 1.,):
    net.eval()
    valid_loss  = 0.
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
                                                        batch_features.to(device),
                                                        reconstruction.to(device),
                                                        recon_loss_func,
                                                        )
            recon_losses += recon_loss
            ## KLD loss
            kld_loss        = compute_kl_divergence(mu, log_var)
            kld_losses += beta * kld_loss
            # backpropagation
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
                        image_loss_func:nn.Module           = nn.NLLLoss(),
                        recon_loss_func:nn.Module           = nn.MSELoss(),
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