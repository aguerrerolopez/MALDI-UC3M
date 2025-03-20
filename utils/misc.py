import numpy as np
import torch
import matplotlib.pyplot as plt


def evaluation(test_loader, name=None, model_best=None, epoch=None):
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
        loss_t, RE_t, KL_t = model_best.forward(test_batch, reduction='sum')
        loss = loss + loss_t.item()
        RE = RE + RE_t.item()
        KL = KL + KL_t.item()
        N = N + test_batch.shape[0]
    loss = loss / N
    RE = RE / N
    KL = KL / N

    if epoch is None:
        print(f'FINAL LOSS: nll={loss}')
        print(f'FINAL RE: {RE}')
        print(f'FINAL KL: {KL}')
    else:
        print(f'Epoch: {epoch}, val nll={loss}')
        print(f'Epoch: {epoch}, val RE: {RE}')
        print(f'Epoch: {epoch}, val KL: {KL}')

    return loss, RE, KL

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
        plottable_image = np.reshape(x[i], (8, 8))
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
    x = next(iter(data_loader)).detach().numpy()

    # GENERATIONS-------
    model_best = torch.load(name + '.model', weights_only=False)
    model_best.eval()

    num_x = 4
    num_y = 4
    x = model_best.sample(num_x * num_y)
    x = x.detach().numpy()

    fig, ax = plt.subplots(num_x, num_y)
    for i, ax in enumerate(ax.flatten()):
        plottable_image = np.reshape(x[i], (8, 8))
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

def training(name, max_patience, num_epochs, model, optimizer, training_loader, val_loader):
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
        for indx_batch, batch in enumerate(training_loader):
            if hasattr(model, 'dequantization'):
                if model.dequantization:
                    batch = batch + torch.rand(batch.shape)
            loss, _, _ = model.forward(batch)

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

        # Validation
        loss_val, RE, KL = evaluation(val_loader, model_best=model, epoch=e)
        nll_val.append(loss_val)  # save for plotting
        RE_val.append(RE) # save for plotting
        KL_val.append(KL) # save for plotting

        if e == 0:
            print('saved!')
            torch.save(model, name + '.model')
            best_nll = loss_val
        else:
            if loss_val < best_nll:
                print('saved!')
                torch.save(model, name + '.model')
                best_nll = loss_val
                patience = 0

                samples_generated(name, val_loader, extra_name="_epoch_" + str(e))
            else:
                patience = patience + 1

        if patience > max_patience:
            break

    nll_val = np.asarray(nll_val)
    RE_val = np.asarray(RE_val)
    KL_val = np.asarray(KL_val)

    return nll_val, RE_val, KL_val