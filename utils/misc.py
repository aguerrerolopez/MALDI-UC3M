import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.losses import loss_function

def plot_train_val_curves(name, train_data, val_data, title="_NLL_RE_KL", legend=['NLL', 'RE', 'KL']):

    assert len(train_data) == len(val_data) == 3, "train_data and val_data must contain as many elements as legend values (default: NLL, RE, KL)."

    nll_curve_train, RE_curve_train, KL_curve_train = train_data
    nll_curve_val, RE_curve_val, KL_curve_val = val_data

    plt.figure(figsize=(10, 6))

    # Colors: NLL = blue, RE = orange, KL = green
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    epochs = np.arange(len(nll_curve_train))

    # Plot training curves
    plt.plot(epochs, nll_curve_train, linestyle='-', color=colors[0], label=legend[0] + ' (train)')
    plt.plot(epochs, RE_curve_train, linestyle='-', color=colors[1], label=legend[1] + ' (train)')
    plt.plot(epochs, KL_curve_train, linestyle='-', color=colors[2], label=legend[2] + ' (train)')

    # Plot validation curves
    plt.plot(epochs, nll_curve_val, linestyle='--', color=colors[0], label=legend[0] + ' (val)')
    plt.plot(epochs, RE_curve_val, linestyle='--', color=colors[1], label=legend[1] + ' (val)')
    plt.plot(epochs, KL_curve_val, linestyle='--', color=colors[2], label=legend[2] + ' (val)')

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(name + title + ".pdf", bbox_inches='tight')
    plt.close()

def early_stopping(epoch, nll_val, best_nll, patience, max_patience, model, name, path, saving='best'):
    """
    Early stopping function
    :param epoch: current epoch
    :param nll_val: negative log-likelihood values
    :param best_nll: best negative log-likelihood value
    :param patience: current patience
    :param max_patience: maximum patience
    :return: updated patience and best_nll
    """
    if epoch == 0 or (nll_val < best_nll):
        saved_path = os.path.join(path, f"{name}_bestmodel.pth") if saving == 'best' else os.path.join(path, f"{name}_epoch_{epoch}.pth")
        torch.save(model.state_dict(), saved_path)
        print("saved!")
        best_nll = nll_val
        patience = 0
    else:
        patience += 1

    if patience > max_patience:
        return True, best_nll, patience, saved_path

    return False, best_nll, patience, saved_path

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0.0
    RE_vals = 0.0
    KL_vals = 0.0

    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss, rec, kl = loss_function(recon_batch, data, mu, logvar, model.loss_mode)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        RE_vals += rec.item()
        KL_vals += kl.item()
        
        if batch_idx % 100 == 0:
            print(f"Epoch {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] Loss: {loss.item() / len(data):.4f}  RECON: {rec.item() / len(data):.4f}  KL: {kl.item() / len(data):.4f}")
    print(f"====> Epoch {epoch} Average loss: {train_loss / len(train_loader.dataset):.4f}  RECON: {RE_vals / len(train_loader.dataset):.4f}  KL: {KL_vals / len(train_loader.dataset):.4f}")

    return train_loss / len(train_loader.dataset), RE_vals / len(train_loader.dataset), KL_vals / len(train_loader.dataset)

def evaluate(test_loader, name=None, model=None, epoch=None, device="cpu"):

    if model is None:
        assert name is not None, "If no model is passed, 'name' must be given to load the model."
        model = torch.load(name + '.model', map_location=device)

    model.to(device)
    model.eval()

    total_loss = 0.0
    total_RE = 0.0
    total_KL = 0.0

    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(test_loader):
            data = data.to(device)

            recon_batch, mu, logvar = model(data)
            loss, rec, kl = loss_function(recon_batch, data, mu, logvar, model.loss_mode)

            total_loss += loss.item()
            total_RE += rec.item()
            total_KL += kl.item()

    avg_nll = total_loss / len(test_loader.dataset)
    avg_RE = total_RE / len(test_loader.dataset)
    avg_KL = total_KL / len(test_loader.dataset)

    if epoch is not None:
        print(f"Epoch {epoch} VALIDATION → NLL: {avg_nll:.4f} | RE: {avg_RE:.4f} | KL: {avg_KL:.4f}")
    else:
        print(f"FINAL VALIDATION → NLL: {avg_nll:.4f} | RE: {avg_RE:.4f} | KL: {avg_KL:.4f}")

    return avg_nll, avg_RE, avg_KL