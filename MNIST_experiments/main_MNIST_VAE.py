import sys
import os
import torch
import time
import numpy as np
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torchvision import datasets, transforms
from pytorch_model_summary import summary

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.VAE import VAE
from utils.misc import evaluation, samples_real, plot_curve, training

### INITIALIZE DATALOADERS

data_name = 'MNIST'
name = 'vae'
result_dir = f'results/{data_name}_{name}_{time.strftime("%Y%m%d_%H%M%S")}/'
os.makedirs(result_dir, exist_ok=True)

transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])

# Load full training dataset
full_train_data = datasets.MNIST(root="./data", train=True, transform=transform, download=True)

# Split into train and validation sets (90% train, 10% validation)
valid_size = 0.1  # 10% for validation
num_train = len(full_train_data)
split = int(np.floor(valid_size * num_train))
train_data, val_data = random_split(full_train_data, [num_train - split, split])

# Load test set
test_data = datasets.MNIST(root="./data", train=False, transform=transform, download=True)

training_loader = DataLoader(train_data, batch_size=64, shuffle=True)
val_loader = DataLoader(val_data, batch_size=64, shuffle=False)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

### HYPERPARAMS

D = 784   # MNIST images are 28x28 pixels
L = 16  # Latent space size
M = 256  # Number of neurons in hidden layers

lr = 5e-4  # Learning rate
num_epochs = 1000  # Epochs to train
max_patience = 20  # Early stopping

### INITIALIZE VAE

likelihood_type = 'bernoulli'
num_vals = 1  # Bernoulli for binary images

encoder = nn.Sequential(
    nn.Linear(D, M), nn.ReLU(),
    nn.Linear(M, M), nn.ReLU(),
    nn.Linear(M, 2 * L))

decoder = nn.Sequential(
    nn.Linear(L, M), nn.ReLU(),
    nn.Linear(M, M), nn.ReLU(),
    nn.Linear(M, num_vals * D))  # Output matches input size 

# Initialize model
model = VAE(encoder_net=encoder, decoder_net=decoder, num_vals=num_vals, L=L, likelihood_type=likelihood_type)

# Print model summary
print("ENCODER:\n", summary(encoder, torch.zeros(1, D), show_input=False, show_hierarchical=False))
print("\nDECODER:\n", summary(decoder, torch.zeros(1, L), show_input=False, show_hierarchical=False))

### TRAINING

optimizer = torch.optim.Adamax([p for p in model.parameters() if p.requires_grad == True], lr=lr)

nll_val, RE_val, KL_val = training(name=result_dir + name, max_patience=max_patience, 
                                   num_epochs=num_epochs, model=model, optimizer=optimizer,
                                   training_loader=training_loader, val_loader=val_loader)

test_loss, test_RE, test_KL = evaluation(name=result_dir + name, test_loader=test_loader)

with open(result_dir + name + '_test_loss.txt', "w") as f:
    f.write(f"NLL: {test_loss}\nRE: {test_RE}\nKL: {test_KL}\n")

# samples_real(result_dir + name, test_loader)
plot_curve(result_dir + name, [nll_val, RE_val, KL_val], title='_NLL_RE_KL', legend=['NLL', 'RE', 'KL'])