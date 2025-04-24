import os
import sys
import torch
import time
import numpy as np
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.bottlenecks import MLP
from models.AE_VAE import VAE
from utils.losses import loss_function
from utils.misc import plot_curve 

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

def main():

    # ------------------------------
    # 1) SETUP: data, hyperparams
    # ------------------------------

    data_name = 'MNIST'
    name = 'mlp_vae'
    result_dir = f'results/{data_name}_{name}_{time.strftime("%Y%m%d_%H%M%S")}/'
    os.makedirs(result_dir, exist_ok=True)

    # Set device and hyperparameters.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 128
    epochs = 10
    patience = 10
    learning_rate = 1e-3
    loss_mode = 'bce'  # Change to 'mse' or 'gaussian' if desired.

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1))  # Flatten from 28x28 => 784
    ])

    # MNIST dataset and DataLoader.
    # Load full training dataset
    full_train_data = datasets.MNIST(root="./data", train=True, transform=transform, download=True)

    # Split into train and validation sets (90% train, 10% validation)
    valid_size = 0.1  # 10% for validation
    num_train = len(full_train_data)
    split = int(np.floor(valid_size * num_train))
    train_data, val_data = random_split(full_train_data, [num_train - split, split])

    # Test data
    test_data = datasets.MNIST(root="./data", train=False, transform=transform, download=True)

    # DataLoader for training, validation, and test sets.
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)


    # ------------------------------
    # 2) DEFINE MODELS
    # ------------------------------

    # Define the encoder and decoder networks.
    bottleneck = MLP() # This can be replaced with any other bottleneck architecture.
    encoder_bot = bottleneck.encoder
    decoder_bot = bottleneck.decoder

    # Instantiate the model, optimizer.
    model = VAE(encoder_bot, decoder_bot, loss_mode=loss_mode).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # ------------------------------
    # 3) TRAIN
    # ------------------------------

    nll_curve_train = []
    nll_curve_val = []
    RE_curve_train = []
    RE_curve_val = []
    KL_curve_train = []
    KL_curve_val = []

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        nll, re, kl = train(model, device, train_loader, optimizer, epoch)
        nll_curve_train.append(nll)
        RE_curve_train.append(re)
        KL_curve_train.append(kl)

        # Validation
        nll_val, re_val, kl_val = evaluate(val_loader, model=model, epoch=epoch, device=device)
        # # Early stopping check
        # best_val_loss, patience_counter, stop = early_stopping(nll, best_val_loss, patience_counter, patience)
        # if stop:
        #     print("Early stopping triggered.")
        #     break

        # Save model
        torch.save(model.state_dict(), os.path.join(result_dir, f"{name}_epoch_{epoch}.pth"))

    # plot_curve(result_dir + name, [nll_curve, RE_curve, KL_curve], title='_NLL_RE_KL', legend=['NLL', 'RE', 'KL'])

if __name__ == "__main__":
    main()
