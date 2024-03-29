#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 13:52:14 2022

@author:  AntixK

https://github.com/AntixK/PyTorch-VAE
I didn't know spyder has a quick documentation feature
"""
import torch
from torchvision          import transforms
from utils_deep           import (candidates,
                                  define_type
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
                                nn.BatchNorm1d(output_units,),
                                output_activation,
                                dropout)
        else:
            hidden_layer = nn.Sequential(
                                latent_layer,
                                nn.BatchNorm1d(output_units),
                                dropout)
        return hidden_layer
    elif layer_type == 'recurrent':
        raise NotImplementedError
    else:
        raise NotImplementedError

def CNN_feature_extractor(pretrained_model_name,
                          retrain_encoder:bool,
                          in_shape:Tuple = (1,3,128,128),
                          device = 'cpu',
                          ):
    """
    

    Parameters
    ----------
    pretrained_model_name : TYPE
        DESCRIPTION.
    retrain_encoder : bool
        DESCRIPTION.
    in_shape : Tuple, optional
        DESCRIPTION. The default is (1,3,128,128).
    device : str or torch.device, optional
        DESCRIPTION. The default is 'cpu'.

    Returns
    -------
    in_features : int
        DESCRIPTION.
    feature_extractor : nn.Module
        DESCRIPTION.

    """
    pretrained_model       = candidates(pretrained_model_name)
    # freeze the pretrained model
    if not retrain_encoder:
        for params in pretrained_model.parameters():
            params.requires_grad = False
    # get the dimensionof the CNN features
    if define_type(pretrained_model_name) == 'simple':
        in_features                = nn.AdaptiveAvgPool2d((1,1))(
                                            pretrained_model.features(
                                                    torch.rand(*in_shape))).shape[1]
        feature_extractor          = easy_model(pretrained_model = pretrained_model,).to(device)
    elif define_type(pretrained_model_name) == 'resnet':
        in_features                = pretrained_model.fc.in_features
        feature_extractor          = resnet_model(pretrained_model = pretrained_model,).to(device)
    return in_features,feature_extractor

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
                 output_activation              = nn.Tanh(),
                 latent_units                   = 300,
                 latent_activation              = nn.Tanh(),
                 in_channels:int                = 3,
                 in_shape:Tuple                 = (1,3,128,128),
                 layer_type:str                 = 'linear',
                 device                         = 'cpu',
                 hidden_dims:List               = None,
                 retrain_encoder:bool           = False,
                 multi_hidden_layer:bool        = False,
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
        latent_units:int, dimesnion of the latent layers (mu,log_var,z)
        latent_activation: nn.Module, activation for the latent layers
        output_activation: nn.Module, activation function of the reconstruction layer
        in_channels:int, number of channels of the input image
        in_shape:Tuple, the dimension of the inpu images (1,n_channels, height, width)
        layer_type:str, type of the hidden layers
        device:str or torch.device,
        hidden_dims:List, the list of hidden dimensions of the decoder
        retrain_encoder:bool, unfreeze the CNN backbone during training
        multi_hidden_layer:bool, if True, we have more than one dense layer
            before the `mu` and `log_var` layers
        
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
        self.hidden_units                   = hidden_units
        self.hidden_activation              = hidden_activation
        self.hidden_dropout                 = hidden_dropout
        self.output_activation              = output_activation
        self.latent_units                   = latent_units
        self.latent_activation              = latent_activation
        self.layer_type                     = layer_type
        self.device                         = device
        self.in_channels                    = in_channels
        self.hidden_dims                    = hidden_dims
        self.multi_hidden_layer             = multi_hidden_layer
        
        # for output channels of the decoder
        if self.hidden_dims == None:
            self.hidden_dims = [self.latent_units,128,64,32,16]
        
        # Build Encoder
        in_features,feature_extractor   = CNN_feature_extractor(
                                            pretrained_model_name   = pretrained_model_name,
                                            retrain_encoder         = retrain_encoder,
                                            in_shape                = in_shape,
                                            device                  = device,
                                                               )
        self.in_features                = in_features
        feature_extractor               = feature_extractor.to(device)
        if self.multi_hidden_layer:
            # we add more dense layers after the CNN layer
            encoder = []
            encoder.append(feature_extractor)
            encoder.append(create_hidden_layer(
                                            layer_type          = self.layer_type,
                                            input_units         = self.in_features,
                                            output_units        = self.hidden_dims[0],
                                            output_activation   = self.hidden_activation,
                                            output_dropout      = self.hidden_dropout,
                                            device              = self.device,
                                            ).to(device)
                                )
            for input_units,output_units in zip(self.hidden_dims[:-1],self.hidden_dims[1:]):
                encoder.append(create_hidden_layer(
                                            layer_type          = self.layer_type,
                                            input_units         = input_units,
                                            output_units        = output_units,
                                            output_activation   = self.hidden_activation,
                                            output_dropout      = self.hidden_dropout,
                                            device              = self.device,
                                            ).to(device)
                                    )
            self.encoder = nn.Sequential(*encoder).to(device)
            ## the mu layer
            self.mu_layer                       = create_hidden_layer(
                                                    layer_type          = self.layer_type,
                                                    input_units         = self.hidden_dims[-1],
                                                    output_units        = self.latent_units,
                                                    output_activation   = self.latent_activation,
                                                    output_dropout      = self.hidden_dropout,
                                                    device              = self.device,
                                                    ).to(self.device)
            ## the log_var layer
            self.log_var_layer                  = create_hidden_layer(
                                                    layer_type          = self.layer_type,
                                                    input_units         = self.hidden_dims[-1],
                                                    output_units        = self.latent_units,
                                                    output_activation   = self.latent_activation,
                                                    output_dropout      = self.hidden_dropout,
                                                    device              = self.device,
                                                    ).to(self.device)
            # Build Decoder
            hidden_dims = self.hidden_dims.copy()
            hidden_dims.reverse()
            modules = [nn.Sequential(
                                nn.ConvTranspose2d(latent_units,
                                                   hidden_dims[0],
                                                   kernel_size      = 3,
                                                   stride           = 2,
                                                   padding          = 1,
                                                   output_padding   = 1,
                                                   ),
                                nn.BatchNorm2d(hidden_dims[0]),
                                nn.LeakyReLU())
                ]
            for ii in range(len(hidden_dims) - 1):
                modules.append(
                        nn.Sequential(
                                nn.ConvTranspose2d(hidden_dims[ii],
                                                   hidden_dims[ii + 1],
                                                   kernel_size      = 3,
                                                   stride           = 2,
                                                   padding          = 1,
                                                   output_padding   = 1,
                                                   ),
                                nn.BatchNorm2d(hidden_dims[ii + 1]),
                                nn.LeakyReLU()
                                    )
                            )
            self.decoder                        = nn.Sequential(*modules).to(self.device)
            self.final_layer                    = nn.Sequential(
                                                    nn.ConvTranspose2d(hidden_dims[-1],
                                                                       hidden_dims[-1],
                                                                       kernel_size      = 3,
                                                                       stride           = 2,
                                                                       padding          = 1,
                                                                       output_padding   = 1,
                                                                       ),
                                                    nn.BatchNorm2d(hidden_dims[-1]),
                                                    nn.LeakyReLU(),
                                                    nn.Conv2d(hidden_dims[-1],
                                                              out_channels  = 3,
                                                              kernel_size   = 3,
                                                              padding       = 1,
                                                              ),
                                                    self.output_activation
                                                    ).to(self.device)
            
        else:
            # we directly connect the CNN to mu and log_var
            ## the mu layer
            self.mu_layer                       = create_hidden_layer(
                                                    layer_type          = self.layer_type,
                                                    input_units         = self.in_features,
                                                    output_units        = self.hidden_units,
                                                    output_activation   = self.latent_activation,
                                                    output_dropout      = self.hidden_dropout,
                                                    device              = self.device,
                                                    ).to(self.device)
            ## the log_var layer
            self.log_var_layer                  = create_hidden_layer(
                                                    layer_type          = self.layer_type,
                                                    input_units         = self.in_features,
                                                    output_units        = self.hidden_units,
                                                    output_activation   = self.latent_activation,
                                                    output_dropout      = self.hidden_dropout,
                                                    device              = self.device,
                                                    ).to(self.device)
            self.encoder                        = feature_extractor.to(device)
            # Build Decoder
            modules = []
            hidden_dims = self.hidden_dims.copy()
            for ii in range(len(hidden_dims) - 1):
                modules.append(
                        nn.Sequential(
                                nn.ConvTranspose2d(hidden_dims[ii],
                                                   hidden_dims[ii + 1],
                                                   kernel_size      = 3,
                                                   stride           = 2,
                                                   padding          = 1,
                                                   output_padding   = 1,
                                                   ),
                                nn.BatchNorm2d(hidden_dims[ii + 1]),
                                nn.LeakyReLU()
                                    )
                            )
            self.decoder                        = nn.Sequential(*modules).to(self.device)
            self.final_layer                    = nn.Sequential(
                                                    nn.ConvTranspose2d(hidden_dims[-1],
                                                                       hidden_dims[-1],
                                                                       kernel_size      = 3,
                                                                       stride           = 2,
                                                                       padding          = 1,
                                                                       output_padding   = 1,
                                                                       ),
                                                    nn.BatchNorm2d(hidden_dims[-1]),
                                                    nn.LeakyReLU(),
                                                    nn.Conv2d(hidden_dims[-1],
                                                              out_channels  = 3,
                                                              kernel_size   = 3,
                                                              padding       = 1,
                                                              ),
                                                    self.output_activation
                                                    ).to(self.device)
        
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
        extracted_features  = self.encoder(x)
        mu                  = self.mu_layer(extracted_features)
        log_var             = self.log_var_layer(extracted_features)
        z                   = self.reparameterize(mu, log_var)
        z                   = z.view(-1,z.shape[1],1,1)
        conv_transpose      = self.decoder(z)
        reconstruction      = self.final_layer(conv_transpose)
        return reconstruction,extracted_features,z,mu,log_var

class simple_classifier(nn.Module):
    def __init__(self,
                 pretrained_model_name:str,
                 hidden_units:int,
                 hidden_activation:nn.Module,
                 hidden_dropout:float,
                 output_units:int,
                 output_activation:nn.Module,
                 hidden_dims:List   = [],
                 in_shape:Tuple     = (1,3,128,128),
                 layer_type:str     = 'linear',
                 device:str         = 'cpu',
                 retrain_encoder    = False,
                 ) -> None:
        super(simple_classifier,self).__init__()
        
        torch.manual_seed(12345)
        self.pretrained_model_name  = pretrained_model_name
        self.hidden_units           = hidden_units
        self.hidden_activation      = hidden_activation
        self.hidden_dropout         = hidden_dropout
        self.output_units           = output_units
        self.output_activation      = output_activation
        self.hidden_dims            = hidden_dims
        self.layer_type             = layer_type
        self.device                 = device
        torch.manual_seed(12345)
        in_features,feature_extractor  = CNN_feature_extractor(
                                            pretrained_model_name   = pretrained_model_name,
                                            retrain_encoder         = retrain_encoder,
                                            in_shape                = in_shape,
                                            device                  = device,
                                                               )
        self.in_features               = in_features
        self.feature_extractor         = feature_extractor.to(device)
        # hidden layer
        if len(hidden_dims) == 0:
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
        else:
            hidden_layer = []
            hidden_layer.append(create_hidden_layer(
                                            layer_type          = self.layer_type,
                                            input_units         = self.in_features,
                                            output_units        = self.hidden_dims[0],
                                            output_activation   = self.hidden_activation,
                                            output_dropout      = self.hidden_dropout,
                                            device              = self.device,
                                            ).to(device)
                                )
            for input_units,output_units in zip(self.hidden_dims[:-1],self.hidden_dims[1:]):
                hidden_layer.append(create_hidden_layer(
                                            layer_type          = self.layer_type,
                                            input_units         = input_units,
                                            output_units        = output_units,
                                            output_activation   = self.hidden_activation,
                                            output_dropout      = self.hidden_dropout,
                                            device              = self.device,
                                            ).to(device)
                                    )
            self.hidden_layer = nn.Sequential(*hidden_layer).to(device)
            # output layer
            self.output_layer = nn.Sequential(
                                            nn.Linear(self.hidden_dims[-1],
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

class vae_classifier(BaseVAE):
    def __init__(self,
                 pretrained_model_name:str      = 'vgg19',
                 hidden_units:int               = 300,
                 hidden_activation:nn.Module    = nn.ReLU(),
                 hidden_dropout:float           = 0.,
                 output_units:int               = 10,
                 vae_output_activation          = nn.Tanhshrink(),
                 clf_output_activation          = nn.Softmax(dim = -1),
                 latent_units                   = 300,
                 latent_activations             = [nn.Tanh(),nn.ReLU()],
                 in_channels:int                = 3,
                 in_shape:Tuple                 = (1,3,128,128),
                 layer_type:str                 = 'linear',
                 device                         = 'cpu',
                 hidden_dims:List               = None,
                 retrain_encoder:bool           = False,
                 multi_hidden_layer:bool        = False,
                 ) -> None:
        """
        CNN --> hidden layer | --> mu ------> | --> decoder
                             | --> log_var -> | --> decoder
                             | --> y_pred

        Parameters
        ----------
        pretrained_model_name : str, optional
            CNN backnone model. The default is 'vgg19'.
        hidden_units : int, optional
            This is used when only 1 hidden layer for the image classifier. The default is 300.
        hidden_activation : nn.Module, optional
            This is used when only 1 hidden layer for the image classifier. The default is nn.ReLU().
        hidden_dropout : float, optional
            This is used when only 1 hidden layer for the image classifier.. The default is 0..
        output_units : int, optional
            The output dimension of the image classifier. The default is 10.
        vae_output_activation : nn.Module, optional
            activation funtion of the reconstruction. The default is nn.Tanhshrink().
        clf_output_activation : nn.Module, optional
            activation function for the image classifier. The default is nn.Softmax(dim = -1).
        latent_units : TYPE, optional
            Dimension of the mu and log_var layers. The default is 300.
        latent_activations : List of nn.Module
            activation functions for the mu and log_var layers. The default is nn.Tanh().
        in_channels : int, optional
            DESCRIPTION. The default is 3.
        in_shape : Tuple, optional
            DESCRIPTION. The default is (1,3,128,128).
        layer_type : str, optional
            DESCRIPTION. The default is 'linear'.
        device : TYPE, optional
            DESCRIPTION. The default is 'cpu'.
        hidden_dims : List, optional
            This is used for multilayer classifier and the decoder. The default is None.
        retrain_encoder : bool, optional
            Whether we want to retrained the CNN backbone layers. The default is False.
        multi_hidden_layer : bool, optional
            To build multilayer classifier. The default is False.

        Returns
        -------
        None
            DESCRIPTION.

        """
        super(vae_classifier,self).__init__()
        torch.manual_seed(12345)
        self.pretrained_model_name          = pretrained_model_name
        self.hidden_units                   = hidden_units
        self.hidden_activation              = hidden_activation
        self.hidden_dropout                 = hidden_dropout
        self.output_units                   = output_units
        self.vae_output_activation          = vae_output_activation
        self.clf_output_activation          = clf_output_activation
        self.latent_units                   = latent_units
        self.latent_activations             = latent_activations
        self.layer_type                     = layer_type
        self.device                         = device
        self.in_channels                    = in_channels
        self.hidden_dims                    = hidden_dims
        self.multi_hidden_layer             = multi_hidden_layer
        self.in_shape                       = in_shape
        self.retrain_encoder                = retrain_encoder
        # in the dataloader, we convert the images to [0,1] using ToTensor()
        # and then we normalize the images using Normalizer()
        # So, here we could rescale the reconstructed images using Sigmoid()
        # and then we normalize them as we did in the dataloader
        # self.normalize                      = transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                                                            std=[0.229, 0.224, 0.225])
        
        # CNN feature extractor
        torch.manual_seed(12345)
        in_features,feature_extractor  = CNN_feature_extractor(
                                            pretrained_model_name   = self.pretrained_model_name,
                                            retrain_encoder         = self.retrain_encoder,
                                            in_shape                = self.in_shape,
                                            device                  = self.device,
                                                               )
        self.in_features = in_features
        feature_extractor = feature_extractor.to(self.device)
        # vae
        if self.multi_hidden_layer:
            print('more dense layers')
            # classifier
            ## hidden layers
            hidden_layer = []
            hidden_layer.append(create_hidden_layer(
                                            layer_type          = self.layer_type,
                                            input_units         = self.in_features,
                                            output_units        = self.hidden_dims[0],
                                            output_activation   = self.hidden_activation,
                                            output_dropout      = self.hidden_dropout,
                                            device              = self.device,
                                            ).to(device)
                                )
            for input_units,output_units in zip(self.hidden_dims[:-1],self.hidden_dims[1:]):
                hidden_layer.append(create_hidden_layer(
                                            layer_type          = self.layer_type,
                                            input_units         = input_units,
                                            output_units        = output_units,
                                            output_activation   = self.hidden_activation,
                                            output_dropout      = self.hidden_dropout,
                                            device              = self.device,
                                            ).to(device)
                                    )
            self.hidden_layer                   = nn.Sequential(*hidden_layer).to(self.device)
            ## output layer
            self.output_layer                   = nn.Sequential(
                                                        nn.Linear(self.hidden_dims[-1],
                                                                  self.output_units),
                                                        self.clf_output_activation
                                                        ).to(self.device)
            
            # encoder
            self.encoder                        = feature_extractor.to(self.device)
            ## the mu layer
            self.mu_layer                       = create_hidden_layer(
                                                    layer_type          = self.layer_type,
                                                    input_units         = self.hidden_dims[-1],
                                                    output_units        = self.latent_units,
                                                    output_activation   = self.latent_activations[0],
                                                    output_dropout      = self.hidden_dropout,
                                                    device              = self.device,
                                                    ).to(self.device)
            ## the log_var layer
            self.log_var_layer                  = create_hidden_layer(
                                                    layer_type          = self.layer_type,
                                                    input_units         = self.hidden_dims[-1],
                                                    output_units        = self.latent_units,
                                                    output_activation   = self.latent_activations[1],
                                                    output_dropout      = self.hidden_dropout,
                                                    device              = self.device,
                                                    ).to(self.device)
            # Build Decoder
            self.decoder                        = create_hidden_layer(
                                                    layer_type          = self.layer_type,
                                                    input_units         = self.hidden_dims[-1],
                                                    output_units        = self.latent_units,
                                                    output_activation   = self.vae_output_activation,
                                                    output_dropout      = self.hidden_dropout,
                                                    device              = self.device,
                                                    ).to(self.device)
            # output layer from the VAE
            self.vae_output_layer               = nn.Sequential(
                                                        nn.Linear(self.latent_unis,
                                                                  self.output_units),
                                                        self.clf_output_activation
                                                        ).to(self.device)
            
        else:
            print('direct connection')
            # classifier
            self.hidden_layer = create_hidden_layer(
                                                layer_type          = self.layer_type,
                                                input_units         = self.in_features,
                                                output_units        = self.hidden_units,
                                                output_activation   = self.hidden_activation,
                                                output_dropout      = self.hidden_dropout,
                                                device              = self.device,
                                                ).to(self.device)
            # output layer
            self.output_layer = nn.Sequential(
                                            nn.Linear(self.hidden_units,
                                                      self.output_units),
                                            self.clf_output_activation
                                            ).to(self.device)
            # we directly connect the CNN to mu and log_var
            ## the mu layer
            self.mu_layer                       = create_hidden_layer(
                                                    layer_type          = self.layer_type,
                                                    input_units         = self.hidden_units,
                                                    output_units        = self.latent_units,
                                                    output_activation   = self.latent_activations[0],
                                                    output_dropout      = self.hidden_dropout,
                                                    device              = self.device,
                                                    ).to(self.device)
            ## the log_var layer
            self.log_var_layer                  = create_hidden_layer(
                                                    layer_type          = self.layer_type,
                                                    input_units         = self.hidden_units,
                                                    output_units        = self.latent_units,
                                                    output_activation   = self.latent_activations[1],
                                                    output_dropout      = self.hidden_dropout,
                                                    device              = self.device,
                                                    ).to(self.device)
            self.encoder                        = feature_extractor.to(self.device)
            # Build Decoder
            self.decoder                        = create_hidden_layer(
                                                    layer_type          = self.layer_type,
                                                    input_units         = self.latent_units,
                                                    output_units        = self.hidden_units,
                                                    output_activation   = self.vae_output_activation,
                                                    output_dropout      = self.hidden_dropout,
                                                    device              = self.device,
                                                    ).to(self.device)
            # output layer from the VAE
            self.vae_output_layer               = nn.Sequential(
                                                        nn.Linear(self.hidden_units,
                                                                  self.output_units),
                                                        self.clf_output_activation
                                                        ).to(self.device)
        
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
        
        # z   = mu + std * eps
        vector_size = log_var.size()
        eps = torch.autograd.Variable(torch.FloatTensor(vector_size).normal_()).to(self.device)
        std = log_var.mul(0.5).exp_()
        z = eps.mul(std).add_(mu)
        return z
    
    def forward(self,x:Tensor,) -> List[Tensor]:
        extracted_features      = self.encoder(x)
        hidden_representation   = self.hidden_layer(extracted_features)
        image_category          = self.output_layer(hidden_representation)
        mu                      = self.mu_layer(hidden_representation)
        log_var                 = self.log_var_layer(hidden_representation)
        z                       = self.reparameterize(mu, log_var)
        # z                       = z.view(z.shape[0],z.shape[1],1,1)
        # conv_transpose          = self.decoder(z)
        # reconstruction          = self.normalize(self.final_layer(conv_transpose))
        reconstruction          = self.decoder(z)
        image_category_recon    = self.vae_output_layer(reconstruction)
        return (reconstruction,
                extracted_features,
                z,mu,log_var,
                hidden_representation,
                image_category,image_category_recon)
    