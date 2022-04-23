
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

