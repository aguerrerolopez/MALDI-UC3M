import os
import sys
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.VAE import VAE

torch.serialization.add_safe_globals({'VAE': VAE})  # ← necesaria para evitar el error de cargar el modelo

# ---------- Extract Latent Space ----------
def get_latent_space(model, data_loader):
    latent_vectors = []
    labels = []
    with torch.no_grad():
        for x, y in data_loader:
            mu, log_var = model.encoder.encode(x)
            z = model.encoder.sample(mu_e=mu, log_var_e=log_var)
            latent_vectors.append(z)
            labels.append(y)
    Z = torch.cat(latent_vectors, dim=0).cpu().numpy()
    Y = torch.cat(labels, dim=0).cpu().numpy()
    return Z, Y

# ---------- Plot t-SNE ----------
def visualize_latent_space(latent_vectors, labels, path):
    tsne = TSNE(n_components=2, random_state=42)
    latent_2d = tsne.fit_transform(latent_vectors)
    plt.figure(figsize=(8,6))
    scatter = plt.scatter(latent_2d[:, 0], latent_2d[:, 1], c=labels, cmap='tab10', alpha=0.7, s=10)
    plt.colorbar(scatter, ticks=range(10))
    plt.title("2D t-SNE visualization of VAE latent space")
    plt.xlabel("Latent Dimension 1")
    plt.ylabel("Latent Dimension 2")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    #Save it in the results folder
    plt.savefig(path)

# ---------- Run ----------

def tsne_VAE(model, test_loader, path):
    Z, Y = get_latent_space(model, test_loader)
    visualize_latent_space(Z, Y, path)


# ---------- Settings ----------
data_path = "./data"
batch_size = 64
trial = "MNIST_vae_20250324_114427"
model_path = f"./results/{trial}/vae.model"
saving_path = model_path.split(".model")[0] + '_tsne.png'

# ---------- Transform & DataLoader ----------
transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
test_data = datasets.MNIST(root=data_path, train=False, transform=transform, download=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

model = torch.load(model_path, weights_only=False)
model.eval()

tsne_VAE(model, test_loader, saving_path)
