#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 13:52:14 2022

@author:  AntixK

https://github.com/AntixK/PyTorch-VAE
I didn't know spyder has a quick documentation feature
"""
import torch
from torch import functional as F
from utils_deep import (candidates,
                        define_type,
                        hidden_activation_functions
                        )

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

def create_hidden_layer(
                        layer_type:str,
                        input_units:int,
                        output_units:int,
                        output_dropout:float,
                        output_activation:nn.Module,
                        device,
                        ):
    """
    create a linear hidden layer
    
    Inputs
    ---
    layer_type: str, default = "linear", well, I want to implement recurrent layers
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
    elif layer_type == 'recurrent':
        raise NotImplementedError
    else:
        raise NotImplementedError
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
                 sigma:float        = 0.1,
                 device             = 'cpu',
                 ) -> None:
        super(GaussianNoise,self).__init__()
        self.sigma              = sigma
        self.device             = device

    def forward(self, x:Tensor) -> Tensor:
        with torch.no_grad():
            matrix_size         = x.shape
            if self.sigma is not None:
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
    in_shape: Tuple, input image dimensions (1,n_channels,height,width)

    Outputs
    --------------------
    out: torch.tensor, the pooled CNN features
    """
    def __init__(self,
                 pretrained_model:nn.Module,
                 in_shape = (1,3,128,128),
                 ) -> None:
        super(easy_model,self).__init__()
        torch.manual_seed(12345)
        self.in_features    = nn.AdaptiveAvgPool2d((1,1))(pretrained_model.features(torch.rand(*in_shape))).shape[1]
        avgpool             = nn.AdaptiveAvgPool2d((1,1))
        # print(f'feature dim = {self.in_features}')
        self.features       = nn.Sequential(pretrained_model.features,
                                            avgpool,)
    def forward(self,x:Tensor) -> Tensor:
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
                 pretrained_model:nn.Module,
                 ) -> None:
        super(resnet_model,self).__init__()
        torch.manual_seed(12345)
        avgpool             = nn.AdaptiveAvgPool2d((1,1))
        self.in_features    = pretrained_model.fc.in_features
        res_net             = torch.nn.Sequential(*list(pretrained_model.children())[:-2])
        # print(f'feature dim = {self.in_features}')
        
        self.features       = nn.Sequential(res_net,
                                            avgpool)
        
    def forward(self,x:Tensor) -> Tensor:
        out                 = torch.squeeze(torch.squeeze(self.features(x),3),2)
        return out

###############################################################################
class VanillaVAE(BaseVAE):
    def __init__(self,
                 pretrained_model_name:str      = 'vgg19',
                 hidden_units:int               = 300,
                 hidden_activation:nn.Module    = nn.ReLU(),
                 hidden_dropout:float           = 0.,
                 in_channels:int                = 3,
                 in_shape:Tuple                 = (1,3,128,128),
                 layer_type:str                 = 'linear',
                 device                         = 'cpu',
                 hidden_dims:List               = None,
                 ) -> None:
        super(VanillaVAE,self).__init__()
        """
        Encoder -->|--> mu      | --> z --> Decoder
                   |--> log_var |
        Every "hidden**" is for making layer `mu` and `log_var`
        
        Inputs
        ---
        pretrained_model_name: str, the name of the CNN backbone
        hidden_units:int, dimension of the hidden layer
        hidden_activation:nn.Module, the nonlinear activation of the hidden layer
        hidden_dropout:float, between 0 and 1, the dropout rate of the hidden layer
        in_channels:int, number of channels of the input image
        in_shape:Tuple, the dimension of the inpu images (1,n_channels, height, width)
        layer_type:str, type of the hidden layers
        device:str or torch.device,
        hidden_dims:List, the list of hidden dimensions of the decoder
        
        Outputs
        ---
        reconstruction:torch.tensor, reconstructed image
        hidden_representation: torch.tensor, features extracted by the CNN backbone, and average pooled
        z: torch.tensor, sampled tensor for the decoder
        mu: torch.tensor
        log_var: torch.tensor
        
        """
        
        torch.manual_seed(12345)
        self.pretrained_model_name          = pretrained_model_name
        pretrained_model                    = candidates(pretrained_model_name)
        self.hidden_units                   = hidden_units
        self.hidden_activation              = hidden_activation
        self.hidden_dropout                 = hidden_dropout
        self.layer_type                     = layer_type
        self.device                         = device
        self.in_channels                    = in_channels
        self.hidden_dims                    = hidden_dims
        # freeze the pretrained CNN layers
        for params in pretrained_model.parameters():
            params.requires_grad = False
        # for output channels of the decoder
        if self.hidden_dims == None:
            self.hidden_dims = [self.latent_units,128,64,32,16]
        # Build Encoder
        ## get the dimensionof the CNN features
        if define_type(self.pretrained_model_name) == 'simple':
            self.in_features                = nn.AdaptiveAvgPool2d((1,1))(
                                                pretrained_model.features(
                                                        torch.rand(*in_shape))).shape[1]
            feature_extractor               = easy_model(pretrained_model = pretrained_model,
                                                         ).to(self.device)
        elif define_type(self.pretrained_model_name) == 'resnet':
            self.in_features                = pretrained_model.fc.in_features
            feature_extractor               = resnet_model(pretrained_model = pretrained_model,
                                                           ).to(self.device)
        ## hidden layer
        # self.hidden_layer                   = create_hidden_layer(
        #                                         layer_type          = self.layer_type,
        #                                         input_units         = self.in_features,
        #                                         output_units        = self.hidden_units,
        #                                         output_dropout      = self.hidden_dropout,
        #                                         output_activation   = self.hidden_activation,
        #                                         device              = self.device,
        #                                         ).to(self.device)
        ## the mu layer
        self.mu_layer                       = create_hidden_layer(
                                                layer_type          = self.layer_type,
                                                input_units         = self.in_features,
                                                output_units        = self.hidden_units,
                                                output_activation   = self.hidden_activation,
                                                output_dropout      = self.hidden_dropout,
                                                device              = self.device,
                                                ).to(self.device)
        ## the log_var layer
        self.log_var_layer                  = create_hidden_layer(
                                                layer_type          = self.layer_type,
                                                input_units         = self.in_features,
                                                output_units        = self.hidden_units,
                                                output_activation   = self.hidden_activation,
                                                output_dropout      = self.hidden_dropout,
                                                device              = self.device,
                                                ).to(self.device)
        # self.encoder                        = nn.Sequential(
        #                                         feature_extractor,
        #                                         self.hidden_layer,
        #                                         ).to(self.device)
        self.encoder                        = feature_extractor.to(device)
        
        # Build Decoder
        modules = []
        for ii in range(len(self.hidden_dims) - 1):
            modules.append(
                    nn.Sequential(
                            nn.ConvTranspose2d(self.hidden_dims[ii],
                                               self.hidden_dims[ii + 1],
                                               kernel_size      = 3,
                                               stride           = 2,
                                               padding          = 1,
                                               output_padding   = 1,
                                               ),
                            nn.BatchNorm2d(self.hidden_dims[ii + 1]),
                            nn.LeakyReLU()
                                )
                        )
        self.decoder                        = nn.Sequential(
                                                *modules
                                                ).to(self.device)
        self.final_layer                    = nn.Sequential(
                                                nn.ConvTranspose2d(self.hidden_dims[-1],
                                                                   self.hidden_dims[-1],
                                                                   kernel_size      = 3,
                                                                   stride           = 2,
                                                                   padding          = 1,
                                                                   output_padding   = 1,
                                                                   ),
                                                nn.BatchNorm2d(self.hidden_dims[-1]),
                                                nn.LeakyReLU(),
                                                nn.Conv2d(self.hidden_dims[-1],
                                                          out_channels  = 3,
                                                          kernel_size   = 3,
                                                          padding       = 1,
                                                          ),
                                                nn.Tanh()
                                                ).to(self.device)
    def encode(self,x:Tensor) -> List[Tensor]:
        """
        Inputs
        ---
        x:torch.tensor
        
        Outputs
        ---
        hidden_representaion:torch.tensor,features extracted by the CNN backbone
        mu:torch.tensor,connected to hidden representations
        log_var:torch.tensor,connected to hidden representations
        """
        hidden_representation = self.encoder(x)
        
        #
        mu      = self.mu_layer(hidden_representation)
        log_var = self.log_var_layer(hidden_representation)
        return hidden_representation,mu,log_var

    def decode(self,z:Tensor) -> Tensor:
        """
        Inputs
        ---
        z:torch.tensor, the sampled tensor for the decoder
        
        Outputs
        ---
        output:torch.tensor, the reconsctructed images
        """
        z               = z.view(-1,self.hidden_units,1,1)
        conv_transpose  = self.decoder(z)
        output          = self.final_layer(conv_transpose)
        return output

    def reparameterize(self,mu:Tensor,log_var:Tensor) -> Tensor:
        """
        sample hidden representation from Q(z | x)
        
        Inputs
        ---
        mu:torch.tensor
        log_var:torch.tensor
        
        Outputs
        ---
        z:torch.tensor
        """
        vector_size = log_var.shape
        # independent noise from different dimensions
        dist        = torch.distributions.multivariate_normal.MultivariateNormal(
                            torch.zeros(vector_size[1]),
                            torch.eye(vector_size[1])
                            )
        # sample epsilon from a multivariate Gaussian distribution
        eps         = dist.sample((vector_size[0],)).to(self.device)
        # std = sqrt(exp(log_var))
        std         = torch.sqrt(torch.exp(log_var)).to(self.device)
        # z = mu + std * eps
        z           = mu + std * eps
        return z
    
    def kl_divergence(self,z:Tensor,mu:Tensor,log_var:Tensor,) -> Tensor:
        """
        Q(z|X) has mean of `mu` and std of `exp(log_var)`
        kld = E[log(Q(z|X)) - log(P(z))], where P() is the function P(z|x)
        https://towardsdatascience.com/variational-autoencoder-demystified-with-pytorch-implementation-3a06bee395ed
        
        Inputs
        ---
        z:torch.tensor
        mu:torch.tensor
        log_var:torch.tensor
        
        Outputs
        ---
        KLD_loss:torch.tensor
        """
        KLD_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return KLD_loss
    
    def reconstruction_loss(self,x:Tensor,reconstruct:Tensor) -> Tensor:
        return nn.MSELoss(x,reconstruct)
    
    def forward(self,x:Tensor,) -> List[Tensor]:
        (hidden_representation,
         mu,
         log_var)       = self.encode(x)
        z               = self.reparameterize(mu, log_var)
        reconstruction  = self.decode(z)
        return reconstruction,hidden_representation,z,mu,log_var

class simple_classifier(nn.Module):
    def __init__(self,
                 pretrained_model_name:str,
                 hidden_units:int,
                 hidden_activation:nn.Module,
                 hidden_dropout:float,
                 output_units:int,
                 output_activation:nn.Module,
                 in_shape:Tuple    = (1,3,128,128),
                 layer_type:str    = 'linear',
                 device:str        = 'cpu',
                 ) -> None:
        super(simple_classifier,self).__init__()
        """
        A simple FCNN model
        Extract image features --> hidden representation --> image category
        
        Inputs
        ---
        pretrained_model_name: str,
        hidden_units: int
        hidden_activation: None or nn.Module, torch activation functions
        hidden_dropout: float, between 0 and 1
        output_units: int,
        output_activation: str, sigmoid or softmax
        in_shape: feature extractor input feature shape, default = (1,3,128,128)
        layer_type: str, currently only "linear" is implemented, default = 'linear'
        device: str or torch.device, default = 'cpu'
        
        Outputs
        ---
        image_category: torch.tensor
        hidden_representation: torch.tensor
        """
        
        torch.manual_seed(12345)
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
        self.hidden_layer = create_hidden_layer(
                                        layer_type          = self.layer_type,
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
                                        output_activation
                                        ).to(device)
    def forward(self,x:Tensor) -> Tuple[Tensor]:
        # extract the CNN features
        CNN_features            = self.feature_extractor(x)
        # hidden layer
        hidden_representation   = self.hidden_layer(CNN_features)
        # image category
        image_category          = self.output_layer(hidden_representation)
        return hidden_representation,image_category