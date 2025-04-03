# Trainer.py

import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.losses import RE_log_prob, KL_divergence

class Trainer:
    def __init__(self, ae_model, vae_model, train_loader, val_loader, optimizer, num_epochs=1000, device='cpu'):
        self.ae_model = ae_model    # AE with .encode(x) / .decode(z)
        self.vae_model = vae_model  # VAE with .forward(out) / .sample(size)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.device = device

        self.ae_model.to(self.device)
        self.vae_model.to(self.device)

    def train(self):
        for epoch in tqdm(range(self.num_epochs)):
            train_loss = self._train_one_epoch()
            val_loss = self._validate_one_epoch()
            print(f'Epoch [{epoch+1}/{self.num_epochs}] | Train Loss {train_loss:.4f} | Val Loss {val_loss:.4f}')

    def _train_one_epoch(self):
        self.ae_model.train()
        self.vae_model.train()

        total_loss = 0.0
        total_samples = 0

        for batch in self.train_loader:
            x, _ = batch if isinstance(batch, (list, tuple)) else (batch, None)
            x = x.to(self.device)
            self.optimizer.zero_grad() # zero out gradients

            # 1) AE encoder
            out = self.ae_model.forward_encode(x)
            batch_size = out.size()[0]

            # 2) Pass embedding to VAE
            z, mu_e, log_var_e = self.vae_model.forward(out)
            
            # Compute KL divergence
            log_p_z = self.vae_model.prior.log_prob(z)                        # log p(z)
            log_q_z = self.vae_model.encoder.log_prob(mu_e, log_var_e, z)     # log q(z|x)
            
            KL = KL_divergence(log_p_z, log_q_z, reduction='avg')             # KL(q(z|x) || p(z))
            
            recon_out = self.vae_model.sample(batch_size)

            # 3) AE decoder
            x_recon = self.ae_model.forward_decode(recon_out)

            # 4) Reconstruction Error
            _, RE = RE_log_prob(x, x_recon, reduction='avg') # esto puede cambiar, tomo ahora RE como la log probabilidad de la reconstrucción pero puede ser también una mse u otra específica para maldis

            ELBO = - RE + KL  # ELBO = -RE - KL #TODO check this

            loss = ELBO

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            total_samples += x.size(0)

        return total_loss / total_samples

    def _validate_one_epoch(self):
        self.ae_model.eval()
        self.vae_model.eval()

        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for batch in self.val_loader:
                x, _ = batch if isinstance(batch, (list, tuple)) else (batch, None)

                # 1) AE encoder
                out = self.ae_model.encode(x)
                out_size = out.size()

                # 2) pass embedding to VAE
                z, kl, mu, log_var = self.vae_model.forward(out)
                recon_out = self.vae_model.sample(out_size) #TODO no estoy segura de esta línea

                # 3) AE decoder
                x_recon = self.ae_model.decode(recon_out)

                # 4) Reconstruction Error
                RE = RE_log_prob(x, z, self.vae_model.decoder)
                loss = RE

                total_loss += loss.item()
                total_samples += x.size(0)

        return total_loss / total_samples