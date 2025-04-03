import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_model_summary import summary


# MLP Encoder and Decoder (e.g. for MNIST)
class MLP(nn.Module):
    """
    MLP Encoder and Decoder for MNIST dataset
    Args:
        input_dim: size of the input data (default: 28*28)
        latent_size: size of the latent space (default: 256)
        hidden_size: size of the hidden layers (default: 512)
    Methods:
        encode: forward pass through the encoder
        decode: forward pass through the decoder
        forward: forward pass through the encoder and decoder
    """

    def __init__(self, input_dim=28*28, latent_size=256, hidden_size=512):
        super(MLP, self).__init__()

        self.input_dim = input_dim
        self.latent_size = latent_size

        # Encoder: Fully connected layers
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_size),  # From 784 to 512
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),  # From 512 to 256
            nn.ReLU(),
            nn.Linear(hidden_size // 2, latent_size)  # Output 256 (latent size)
        )

        # Decoder: Fully connected layers
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, hidden_size // 2),  # Latent size to 256
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size),  # 256 to 512
            nn.ReLU(),
            nn.Linear(hidden_size, input_dim),  # Output size matches input_dim (784 for MNIST)
            nn.Sigmoid()  # Output values between 0 and 1 for MNIST
        )

    def forward_encode(self, x):
            # Print summary of the encoder
            print("MLP ENCODER:\n", summary(self.encoder, torch.zeros(1, self.input_dim), show_input=False, show_hierarchical=False))
            return self.encoder(x)
        
    def forward_decode(self, z):
            # Print summary of the decoder
            print("\nMLP DECODER:\n", summary(self.decoder, torch.zeros(1, self.latent_size), show_input=False, show_hierarchical=False))
            return self.decoder(z)
