import sys
import os
import time
import torch
import numpy as np
import torch.nn as nn

from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.bottlenecks import MLP
from models.VAE import VAE  # The "middle" VAE
from utils.Trainer import Trainer  # The trainer class that does MLP -> VAE -> MLP

def main():

    # ------------------------------
    # 1) SETUP: data, hyperparams
    # ------------------------------
    data_name = 'MNIST'
    name = 'mlp_vae'
    result_dir = f'results/{data_name}_{name}_{time.strftime("%Y%m%d_%H%M%S")}/'
    os.makedirs(result_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1))  # Flatten from 28x28 => 784
    ])

    # Load full training dataset
    full_train_data = datasets.MNIST(
        root="./data",
        train=True,
        transform=transform,
        download=True
    )

    # Split into train and validation sets (90% train, 10% validation)
    valid_size = 0.1
    num_train = len(full_train_data)
    split = int(np.floor(valid_size * num_train))
    train_data, val_data = random_split(full_train_data, [num_train - split, split])

    # Load test set
    test_data = datasets.MNIST(
        root="./data",
        train=False,
        transform=transform,
        download=True
    )

    # Create DataLoaders
    training_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

    # ------------------------------
    # 2) DEFINE MODELS
    # ------------------------------
    # MLP model
    D = 784   # MNIST images are 28x28
    embed_dim = 256  # MLP's embedding dimension
    mlp_model = MLP(input_dim=D, latent_size=embed_dim)

    # VAE model (that takes the MLP's embedding dimension as input dimension)
    latent_dim = 32  # e.g. your VAE's internal latent dimension
    vae_model = VAE(D=embed_dim, L=latent_dim)  # So it expects input of size 256, outputs embedding_recon of size 256

    # ------------------------------
    # 3) SETUP OPTIMIZER
    # ------------------------------
    lr = 1e-3
    # combine parameters from both MLP and VAE
    optimizer = torch.optim.Adam(
        list(mlp_model.parameters()) + list(vae_model.parameters()), lr=lr
    )

    # ------------------------------
    # 4) CREATE TRAINER
    # ------------------------------
    trainer = Trainer(
        ae_model=mlp_model,
        vae_model=vae_model,
        train_loader=training_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        num_epochs=20,       # or 100, etc.
        device='cpu'         # or 'cuda'
    )

    # ------------------------------
    # 5) TRAIN
    # ------------------------------
    trainer.train()

    # Optionally: evaluate on test set or do advanced logging
    # e.g. we can do a quick loop:
    test_loss = 0.0
    test_samples = 0
    with torch.no_grad():
        for x, _ in test_loader:
            embedding = mlp_model.encode(x)
            embedding_recon, kl, z, mu, log_var = vae_model(embedding)
            x_recon = mlp_model.decode(embedding_recon)

            # e.g. MSE or BCE
            re = nn.functional.binary_cross_entropy(x_recon, x, reduction='sum')
            kl_sum = kl.sum()

            loss = re + kl_sum
            test_loss += loss.item()
            test_samples += x.size(0)
    test_loss /= test_samples
    print(f"Final Test Loss: {test_loss:.4f}")

if __name__ == "__main__":
    main()