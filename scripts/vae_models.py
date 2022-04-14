#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 13:52:14 2022

@author:  AntixK

https://github.com/AntixK/PyTorch-VAE

"""
import torch


from typing import List, Callable, Union, Any, TypeVar, Tuple
from torch import nn
from abc import abstractmethod

###############################################################################
Tensor = TypeVar('torch.tensor')
###############################################################################
class BaseVAE(nn.Module):
    
    def __init__(self) -> None:
        super(BaseVAE, self).__init__()

    def encode(self, input: Tensor) -> List[Tensor]:
        raise NotImplementedError

    def decode(self, input: Tensor) -> Any:
        raise NotImplementedError

    def sample(self, batch_size:int, current_device: int, **kwargs) -> Tensor:
        raise NotImplementedError

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def forward(self, *inputs: Tensor) -> Tensor:
        pass

    @abstractmethod
    def loss_function(self, *inputs: Any, **kwargs) -> Tensor:
        pass
###############################################################################
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
###############################################################################
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

###############################################################################
class VanillaVAE(BaseVAE):
    def __init__(self,
                 pretrained_model_name,
                  in_channels:int,
                  latent_dim:int,
                  hidden_dims:List = None,
                  ) -> None:
        super(VanillaVAE,self).__init__()
        
        torch.manual_seed(12345)
        self.pretrained_model       = candidates(pretrained_model_name)
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        