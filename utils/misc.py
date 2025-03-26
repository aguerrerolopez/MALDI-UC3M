import numpy as np
import torch
import matplotlib.pyplot as plt

def samples_real(name, test_loader):
    """
    Save real samples as a pdf
    :param name: name of the file
    :param test_loader: test data loader
    :return: None
    """
    # REAL-------
    num_x = 4
    num_y = 4
    x = next(iter(test_loader)).detach().numpy()

    fig, ax = plt.subplots(num_x, num_y)
    for i, ax in enumerate(ax.flatten()):
        plottable_image = np.reshape(x[i], (int(np.sqrt(len(x[i]))), int(np.sqrt(len(x[i])))))
        ax.imshow(plottable_image, cmap='gray')
        ax.axis('off')

    plt.savefig(name+'_real_images.pdf', bbox_inches='tight')
    plt.close()

def samples_generated(name, data_loader, extra_name=''):
    """
    Generate samples from the model and save them as a pdf
    :param name: name of the file
    :param data_loader: data loader
    :param extra_name: extra name for the file
    :return: None
    """

    x = next(iter(data_loader))
    x, _ = x if len(x) == 2 else x
    x = x.detach().numpy() if len(x) == 2 else x.detach().numpy()

    # GENERATIONS-------
    model_best = torch.load(name + '.model', weights_only=False)
    model_best.eval()

    num_x = 4
    num_y = 4
    x = model_best.sample(num_x * num_y)
    x = x.detach().numpy()

    fig, ax = plt.subplots(num_x, num_y)
    for i, ax in enumerate(ax.flatten()):
        plottable_image = np.reshape(x[i], (int(np.sqrt(len(x[i]))), int(np.sqrt(len(x[i])))))
        ax.imshow(plottable_image, cmap='gray')
        ax.axis('off')

    plt.savefig(name + '_generated_images' + extra_name + '.pdf', bbox_inches='tight')
    plt.close()

def plot_curve(name, data, title='_loss_curve', legend=None):
    """
    Plot the negative log-likelihood curve
    :param name (str): path to store the file
    :param data (list): data to plot
    :param title (str): title of the plot
    :param legend (list): legend of the plot
    """
    if not isinstance(data, list):
        data = [data]

    assert len(data) == len(legend), "Data and legend should have the same length"

    for i in range(len(data)):
        plt.plot(np.arange(len(data[i])), data[i], linewidth='3', label=legend[i])
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig(name + title + '.pdf', bbox_inches='tight')
    plt.close()

def training_VAE(name, max_patience, num_epochs, model, optimizer, training_loader, val_loader):
    """
    Training loop
    :param name: name of the file
    :param max_patience: maximum patience
    :param num_epochs: number of epochs
    :param model: model initialization
    :param optimizer: optimizer initialization
    :param training_loader: training data loader
    :param val_loader: validation data loader
    :return nll_val: negative log-likelihood values
    """

    nll_val = []
    RE_val = []
    KL_val = []
    best_nll = 1000.
    patience = 0

    # Main loop
    for e in range(num_epochs):
        # TRAINING
        model.train()
        for batch in training_loader:
            if hasattr(model, 'dequantization'):
                if model.dequantization:
                    batch = batch + torch.rand(batch.shape)
            batch, _ = batch if len(batch) == 2 else batch # for datasets with labels it is necessary to just take the first element (the data)
            loss, _, _ = model.forward(batch)

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

        # Validation
        loss_val, RE, KL = evaluation_VAE(val_loader, model_best=model, epoch=e)
        nll_val.append(loss_val)  # save for plotting
        RE_val.append(RE) # save for plotting
        KL_val.append(KL) # save for plotting

        # Save best model
        if e == 0 or (loss_val < best_nll):
            print("saved!")
            torch.save(model, name + ".model")
            best_nll = loss_val
            patience = 0
            # Generate samples
            samples_generated(name, val_loader, extra_name=f"_epoch_{e}")
        else:
            patience += 1

        if patience > max_patience:
            break

    nll_val = np.asarray(nll_val)
    RE_val = np.asarray(RE_val)
    KL_val = np.asarray(KL_val)

    return nll_val, RE_val, KL_val

def evaluation_VAE(test_loader, name=None, model_best=None, epoch=None):
    """
    Evaluate the model on the test set
    :param test_loader: test data loader
    :param name: name of the file
    :param model_best: model initialization (default: VAE())
    :param epoch: epoch number
    :return: loss
    """
    # EVALUATION
    if model_best is None:
        # load best performing model
        model_best = torch.load(name + '.model', weights_only=False)

    model_best.eval()
    loss = 0.
    RE = 0.
    KL = 0.
    N = 0.
    for indx_batch, test_batch in enumerate(test_loader):
        test_batch, _ = test_batch if len(test_batch) == 2 else test_batch # for datasets with labels it is necessary to just take the first element (the data)
        loss_t, RE_t, KL_t = model_best.forward(test_batch, reduction='sum')
        loss = loss + loss_t.item()
        RE = RE + RE_t.item()
        KL = KL + KL_t.item()
        N = N + test_batch.shape[0]
    loss /= N
    RE /= N
    KL /= N

    if epoch is None:
        print(f'FINAL LOSS: nll={loss}')
        print(f'FINAL RE: {RE}')
        print(f'FINAL KL: {KL}')
    else:
        print(f'Epoch: {epoch}, val nll={loss}')
        print(f'Epoch: {epoch}, val RE: {RE}')
        print(f'Epoch: {epoch}, val KL: {KL}')

    return loss, RE, KL

def training_AE(name, max_patience, num_epochs, model, part, optimizer, training_loader, val_loader, criterion=None):
    """
    Training loop for Autoencoder (MLP).
    """
    best_val_loss = 1e9
    patience = 0

    val_curve = []
    for epoch in range(num_epochs):
        model.train()
        for batch in training_loader:
            batch, _ = batch if isinstance(batch, (list, tuple)) else (batch, None)
            optimizer.zero_grad()

            out = bottleneck.forward_encoder(batch) #bottleneck encoder = MLP, CNN, ...
            out_2 = model.forward(out) #model = VAE, beta-VAE, GAN
            out = bottleneck.forward_decoder(out_2) #bottleneck decoder = MLP, CNN, ...

            #TODO calcular ELBO





            if part == 'encoder':
                out = model.encode(batch)
            elif part == 'decoder':
                out = model.decode(batch)
            else:
                ValueError(f"Part {part} not recognized (only 'encoder0 or 'decoder' supported).")

            loss = criterion(out, batch)

            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch, _ = batch if isinstance(batch, (list, tuple)) else (batch, None)
                if part == 'encoder':
                    out = model.encode(batch)
                elif part == 'decoder':
                    out = model.decode(batch)
                else:
                    ValueError(f"Part {part} not recognized (only 'encoder0 or 'decoder' supported).")

                val_loss += criterion(out, batch).item()
        val_loss /= len(val_loader)
        val_curve.append(val_loss)

        print(f"Epoch [{epoch+1}], AE Val Loss: {val_loss:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            print("saved best AE model!")
            torch.save(model, name + "_AE.model")
            best_val_loss = val_loss
            patience = 0
        else:
            patience += 1

        if patience > max_patience:
            break

    return val_curve

