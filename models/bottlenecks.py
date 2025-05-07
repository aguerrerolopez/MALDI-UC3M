import torch.nn as nn


# MLP Encoder and Decoder (e.g. for MNIST)
class MLP(nn.Module):
    """
    MLP Encoder and Decoder for MNIST dataset
    Args:
        input_dim: size of the input data (default: 28*28)
        latent_size: size of the latent space (default: 256)
        hidden_size: size of the hidden layers (default: 512)
    Methods:
    """

    def __init__(self, input_dim=2000, latent_size=256):
        super(MLP, self).__init__()

        self.input_dim = input_dim
        self.latent_size = latent_size

        # Encoder: Fully connected layers
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, self.latent_size),
            nn.ReLU())

        # Decoder: Fully connected layers
        self.decoder = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.latent_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.input_dim),
            nn.Sigmoid()  # Sigmoid activation for output layer to be between 0 and 1
        )

    # def forward_encode(self, x):
    #         # Print summary of the encoder
    #         # print("MLP ENCODER:\n", summary(self.encoder, torch.zeros(1, self.input_dim), show_input=False, show_hierarchical=False))
    #         return self.encoder(x)
        
    # def forward_decode(self, z):
    #         # Print summary of the decoder
    #         # print("\nMLP DECODER:\n", summary(self.decoder, torch.zeros(1, self.latent_size), show_input=False, show_hierarchical=False))
    #         return self.decoder(z)
