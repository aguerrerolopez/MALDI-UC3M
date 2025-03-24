import sys
import os
import torch
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from torch.utils.data import DataLoader
import torch.nn as nn
from pytorch_model_summary import summary

from MNIST_experiments.Digits import Digits
from models.VAE import VAE
from utils.misc import evaluation, samples_real, plot_curve, training


### INITIALIZE DATALOADERS

data_name = 'digits'
train_data = Digits(mode='train')
val_data = Digits(mode='val')
test_data = Digits(mode='test')

training_loader = DataLoader(train_data, batch_size=64, shuffle=True)
val_loader = DataLoader(val_data, batch_size=64, shuffle=False)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

name = 'vae'
result_dir = f'results/{data_name}_{name}_{time.strftime("%Y%m%d_%H%M%S")}/'
os.makedirs(result_dir, exist_ok=True)

### HYPERPARAMS

D = 64   # input dimension
L = 16  # number of latents
M = 256  # the number of neurons in scale (s) and translation (t) nets

lr = 1e-3 # learning rate
num_epochs = 1000 # max. number of epochs
max_patience = 20 # an early stopping is used, if training doesn't improve for longer than 20 epochs, it is stopped


### INITIALIZE VAE

likelihood_type = 'categorical'

if likelihood_type == 'categorical':
    num_vals = 17
elif likelihood_type == 'bernoulli':
    num_vals = 1

encoder = nn.Sequential(nn.Linear(D, M), nn.LeakyReLU(),
                        nn.Linear(M, M), nn.LeakyReLU(),
                        nn.Linear(M, 2 * L))

decoder = nn.Sequential(nn.Linear(L, M), nn.LeakyReLU(),
                        nn.Linear(M, M), nn.LeakyReLU(),
                        nn.Linear(M, num_vals * D))

# Distribution for the prior from a standard normal (mean 0, std 1)
prior = torch.distributions.MultivariateNormal(torch.zeros(L), torch.eye(L))

# Initialize the model
model = VAE(encoder_net=encoder, decoder_net=decoder, num_vals=num_vals, L=L, likelihood_type=likelihood_type)

# Print the summary (like in Keras)
print("ENCODER:\n", summary(encoder, torch.zeros(1, D), show_input=False, show_hierarchical=False))
print("\nDECODER:\n", summary(decoder, torch.zeros(1, L), show_input=False, show_hierarchical=False))

### TRAINING

optimizer = torch.optim.Adamax([p for p in model.parameters() if p.requires_grad == True], lr=lr)

nll_val, RL_val, KL_val = training(name=result_dir + name, max_patience=max_patience, 
                                   num_epochs=num_epochs, model=model, optimizer=optimizer,
                                   training_loader=training_loader, val_loader=val_loader)

test_loss, test_RE, test_KL = evaluation(name=result_dir + name, test_loader=test_loader)
f = open(result_dir + name + '_test_loss.txt', "w")
f.write(f"NLL: {str(test_loss)}\n")
f.write(f"RE: {str(test_RE)}\n")
f.write(f"KL: {str(test_KL)}\n")
f.close()

samples_real(result_dir + name, test_loader)

plot_curve(result_dir + name, [nll_val, RL_val, KL_val], title='_NLL_RE_KL', legend=['NLL', 'RE', 'KL'])