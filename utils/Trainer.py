# trainer.py

import torch
import tqdm

from utils.losses import RE_log_prob

class Trainer:
    def __init__(self, ae_model, vae_model, train_loader, val_loader, optimizer, num_epochs=1000):
        self.ae_model = ae_model    # AE with .encode(x) / .decode(z)
        self.vae_model = vae_model  # VAE with .forward(embedding)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.num_epochs = num_epochs

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
            self.optimizer.zero_grad() # zero out gradients

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