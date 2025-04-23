import os
import sys
import torch
import time

import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.bottlenecks import MLP
from models.AE_VAE import VAE
from utils.losses import loss_function
from utils.misc import samples_generated, samples_real, plot_curve, early_stopping

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0.0
    RE_vals = []
    KL_vals = []

    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss, rec, kl = loss_function(recon_batch, data, mu, logvar, model.loss_mode)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        RE_vals.append(rec.item())
        KL_vals.append(kl.item())
        
        if batch_idx % 100 == 0:
            print(f"Epoch {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] Loss: {loss.item() / len(data):.4f}  RECON: {rec.item() / len(data):.4f}  KL: {kl.item() / len(data):.4f}")
    print(f"====> Epoch {epoch} Average loss: {train_loss / len(train_loader.dataset):.4f}")

    return train_loss / len(train_loader.dataset), sum(RE_vals)/len(RE_vals), sum(KL_vals)/len(KL_vals)

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
    train_data = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
    test_data = datasets.MNIST(root="./data", train=False, transform=transform, download=True)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)


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

    nll_curve = []
    RE_curve = []
    KL_curve = []
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        nll, RE, KL = train(model, device, train_loader, optimizer, epoch)
        nll_curve.append(nll)
        RE_curve.append(RE)
        KL_curve.append(KL)

        # # Early stopping check
        # best_val_loss, patience_counter, stop = early_stopping(nll, best_val_loss, patience_counter, patience)
        # if stop:
        #     print("Early stopping triggered.")
        #     break

        # Save model
        torch.save(model.state_dict(), os.path.join(result_dir, f"{name}_epoch_{epoch}.pth"))
        # samples_generated(name=result_dir + name, data_loader=test_loader, extra_name=f"_epoch_{epoch}")

    plot_curve(result_dir + name, [nll_curve, RE_curve, KL_curve], title='_NLL_RE_KL', legend=['NLL', 'RE', 'KL'])
    samples_real(result_dir + name, test_loader)

if __name__ == "__main__":
    main()
