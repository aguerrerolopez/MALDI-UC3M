import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.bottlenecks import MLP
from utils.losses import kl_divergence, RE

# Define the VAE model.
class VAE(nn.Module):
    def __init__(self, encoder_bot, decoder_bot, loss_mode='bce'):
        """
        loss_mode can be 'bce', 'mse', or 'gaussian'
        """
        super(VAE, self).__init__()
        self.loss_mode = loss_mode

        # Bottleneck's encoder and decoder
        self.encoder_bot = encoder_bot
        self.decoder_bot = decoder_bot

        # VAE's encoder and decoder
        self.vae_enc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 32*2))

        self.vae_decoder = nn.Sequential(
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, 256))

    def encode(self, x):
        bot = self.encoder_bot(x)
        # Get latent space parameters.
        mu, logvar = self.vae_enc(bot).chunk(2, dim=-1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        # Standard reparameterization trick.
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        vae_dec = self.vae_decoder(z)
        h3 = self.decoder_bot(vae_dec)
        # For BCE loss, it is customary to use a sigmoid output.
        if self.loss_mode == 'bce':
            return torch.sigmoid(h3)
        else:
            return h3

    def forward(self, x):
        # Encode input, reparameterize, then decode.
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

def loss_function(recon_x, x, mu_q, logvar_q, loss_mode, reduction='sum'):
    # Calculate KL divergence loss.
    kl = kl_divergence(mu_q, logvar_q, reduction=reduction)
    
    # Calculate reconstruction error.
    rec_loss = RE(recon_x, x, loss_mode, reduction=reduction)
    
    return rec_loss + kl, rec_loss, kl

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0.0
    for batch_idx, (data, _) in enumerate(train_loader):
        # Flatten the 28x28 image to a 784 vector.
        data = data.view(-1, 784).to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss, rec, kl = loss_function(recon_batch, data, mu, logvar, model.loss_mode)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f"Epoch {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] Loss: {loss.item() / len(data):.4f}  RECON: {rec.item() / len(data):.4f}  KL: {kl.item() / len(data):.4f}")
    print(f"====> Epoch {epoch} Average loss: {train_loss / len(train_loader.dataset):.4f}")

def main():
    # Set device and hyperparameters.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 128
    epochs = 20
    learning_rate = 1e-3
    loss_mode = 'bce'  # Change to 'mse' or 'gaussian' if desired.

    # MNIST dataset and DataLoader.
    train_loader = DataLoader(
        datasets.MNIST('./data', train=True, download=True, transform=transforms.ToTensor()),
        batch_size=batch_size, shuffle=True
    )

    # Define the encoder and decoder networks.
    bottleneck = MLP() # This can be replaced with any other bottleneck architecture.
    encoder_bot = bottleneck.encoder
    decoder_bot = bottleneck.decoder

    # Instantiate the model, optimizer.
    model = VAE(encoder_bot, decoder_bot, loss_mode=loss_mode).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop.
    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, epoch)

if __name__ == "__main__":
    main()
