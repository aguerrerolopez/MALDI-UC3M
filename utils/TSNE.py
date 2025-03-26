import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def get_latent_space_ae(model, data_loader):
    """
    Extracts the latent space for a regular Autoencoder.
    Args:
        model (nn.Module): The Autoencoder model.
        data_loader (DataLoader): DataLoader for the dataset.
        input_size (int): Size of the input data (e.g., 28*28 for MNIST).
    Returns:
        latent_vectors (np.array): Latent space representations.
        labels (np.array): Labels of the dataset.
    """
    latent_vectors = []
    labels = []
    
    with torch.no_grad():
        for x, y in data_loader:
            z = model.encode(x)  # Get latent space representations (AE)
            latent_vectors.append(z)
            labels.append(y)
    
    latent_vectors = torch.cat(latent_vectors, dim=0).cpu().numpy()
    labels = torch.cat(labels, dim=0).cpu().numpy()
    
    return latent_vectors, labels


def get_latent_space_vae(model, data_loader):
    """
    Extracts the latent space for a Variational Autoencoder.
    Args:
        model (nn.Module): The VAE model.
        data_loader (DataLoader): DataLoader for the dataset.
        input_size (int): Size of the input data (e.g., 28*28 for MNIST).
    Returns:
        latent_vectors (np.array): Latent space representations.
        labels (np.array): Labels of the dataset.
    """
    latent_vectors = []
    labels = []
    
    with torch.no_grad():
        for x, y in data_loader:
            mu, log_var = model.encoder.encode(x)
            z = model.encoder.sample(mu_e=mu, log_var_e=log_var)
            latent_vectors.append(z)
            labels.append(y)
    
    latent_vectors = torch.cat(latent_vectors, dim=0).cpu().numpy()
    labels = torch.cat(labels, dim=0).cpu().numpy()
    
    return latent_vectors, labels

def visualize_latent_space(latent_vectors, labels, path):
    """
    Visualizes the latent space using t-SNE.
    Args:
        latent_vectors (np.array): Latent space representations.
        labels (np.array): Labels for each sample.
        path (str): Path to save the t-SNE plot.
    """
    tsne = TSNE(n_components=2, random_state=42)
    latent_2d = tsne.fit_transform(latent_vectors)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(latent_2d[:, 0], latent_2d[:, 1], c=labels, cmap='tab10', alpha=0.7, s=10)
    plt.colorbar(scatter, ticks=range(10))
    plt.title("2D t-SNE visualization of latent space")
    plt.xlabel("Latent Dimension 1")
    plt.ylabel("Latent Dimension 2")
    plt.grid(True)
    plt.tight_layout()

    # Save the t-SNE plot
    plt.savefig(path)
    plt.show()

def TSNE_AE(model, data_loader, save_path=None):
    """
    Visualizes the latent space of an MLP model using t-SNE.
    Args:
        model (MLP): MLP model with an encoder.
        data_loader (DataLoader): DataLoader for the dataset.
        input_size (int): Input size of the data (default: 28*28).
        save_path (str): Path to save the t-SNE plot.
    """
    name = 'tsne_AE.png'
    save_path = f'{save_path}/{name}' if save_path else name
    latent_vectors, labels = get_latent_space_ae(model, data_loader)
    visualize_latent_space(latent_vectors, labels, save_path)

def TSNE_VAE(model, data_loader, save_path=None):
    """
    Visualizes the latent space of a VAE model using t-SNE.
    Args:
        model (VAE): VAE model with an encoder.
        data_loader (DataLoader): DataLoader for the dataset.
        input_size (int): Input size of the data (default: 28*28).
        save_path (str): Path to save the t-SNE plot.
    """
    name = 'tsne_VAE.png'
    save_path = f'{save_path}/{name}' if save_path else name
    latent_vectors, labels = get_latent_space_vae(model, data_loader)
    visualize_latent_space(latent_vectors, labels, save_path)
