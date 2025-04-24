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
from utils.misc import plot_train_val_curves, early_stopping, train, evaluate

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
    epochs = 10
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

    max_patience = 20
    patience = 0
    best_nll = float('inf')

    for epoch in range(1, epochs + 1):
        nll, re, kl = train(model, device, train_loader, optimizer, epoch)
        nll_curve_train.append(nll)
        RE_curve_train.append(re)
        KL_curve_train.append(kl)

        # ------------------------------
        # 4) VALIDATE
        # ------------------------------

        nll_val, re_val, kl_val = evaluate(val_loader, model=model, epoch=epoch, device=device)
        nll_curve_val.append(nll_val)
        RE_curve_val.append(re_val)
        KL_curve_val.append(kl_val)

        # Early stopping check and save best model
        early_stopped, best_nll, patience, saved_path = early_stopping(epoch, nll_val, best_nll, patience, max_patience, model, name, result_dir, saving='epochwise')

        if early_stopped:
            print(f"Early stopping at epoch {epoch} with a loss of {best_nll}.")
            print(f"Best model saved at: {saved_path}")
            break

    train_data = [nll_curve_train, RE_curve_train, KL_curve_train]
    val_data = [nll_curve_val, RE_curve_val, KL_curve_val]

    plot_train_val_curves(result_dir + name, train_data, val_data)

if __name__ == "__main__":
    main()
