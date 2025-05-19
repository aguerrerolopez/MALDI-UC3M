import torch
import torch.nn as nn


class VAE(nn.Module):
    def __init__(self, encoder_bot, decoder_bot, loss_mode='bce', tsne='True'):
        """
        loss_mode can be 'bce', 'mse', or 'gaussian'
        """
        super(VAE, self).__init__()
        self.loss_mode = loss_mode
        self.tsne = tsne

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
        return mu, logvar, bot

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
        mu, logvar, bot = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar, bot, z