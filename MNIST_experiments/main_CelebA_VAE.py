import sys
import os
import torch
import time
import torch.nn as nn

import torchvision.transforms as transforms
from torchvision.datasets import CelebA
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.VAE import VAE
from utils.misc import training, evaluation, samples_real, plot_curve

### INITIALIZE DATALOADERS

data_name = 'CelebA'
name = 'vae'
result_dir = f'results/{data_name}_{name}_{time.strftime("%Y%m%d_%H%M%S")}/'
os.makedirs(result_dir, exist_ok=True)

# Transform: Resize and normalize images to [-1, 1]
image_size = 64
batch_size = 64
transform = transforms.Compose([
    transforms.CenterCrop(178),
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)  # RGB images
])

# Load CelebA dataset
train_dataset = CelebA(root='./data/CelebA', split="train", transform=transform, download=True)
val_dataset = CelebA(root='./data/CelebA', split="valid", transform=transform, download=False)
test_dataset = CelebA(root='./data/CelebA', split="test", transform=transform, download=False)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

### HYPERPARAMS

C = 3  # Number of channels

lr = 1e-3       # learning rate
num_epochs = 50  # Epochs to train
max_patience = 20  # Early stopping

### ENCODER
encoder = nn.Sequential(
    nn.Conv2d(in_channels=C, out_channels=128, kernel_size=4, stride=2, padding=1),  # (64, 64, 3) -> (32, 32, 128)
    nn.ReLU(),
    nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),  # (32, 32, 128) -> (16, 16, 256)
    nn.ReLU(),
    nn.Flatten(),  # (16, 16, 256) -> (65536)
    nn.Linear(in_features=256 * 16 * 16, out_features=256),  # (65536) -> (256)
    nn.ReLU()
)

### DECODER
decoder = nn.Sequential(
    nn.Linear(in_features=256, out_features=256 * 16 * 16),  # (256) -> (65536)
    nn.ReLU(),
    nn.Unflatten(dim=1, unflattened_size=(256, 16, 16)),  # (65536) -> (16, 16, 256)
    nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),  # (16, 16, 256) -> (32, 32, 128)
    nn.ReLU(),
    nn.ConvTranspose2d(in_channels=128, out_channels=C, kernel_size=4, stride=2, padding=1),  # (32, 32, 128) -> (64, 64, 3)
    nn.Sigmoid()
)

### INITIALIZE VAE
likelihood_type = "gaussian"
model = VAE(likelihood_type=likelihood_type)

### TRAINING THE ENCODER
optimizer = torch.optim.Adam(model.parameters(), lr=lr)



optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# Train the model
nll_val, RE_val, KL_val = training(name=result_dir + name,
                                   max_patience=max_patience,
                                   num_epochs=num_epochs,
                                   model=model,
                                   optimizer=optimizer,
                                   training_loader=train_loader,
                                   val_loader=val_loader)

# Evaluate on test set
test_loss, test_RE, test_KL = evaluation(name=result_dir + name, test_loader=test_loader)

with open(result_dir + name + '_test_loss.txt', "w") as f:
    f.write(f"NLL: {test_loss}\nRE: {test_RE}\nKL: {test_KL}\n")

# samples_real(result_dir + name, test_loader)
plot_curve(result_dir + name, [nll_val, RE_val, KL_val], title='_NLL_RE_KL', legend=['NLL', 'RE', 'KL'])