import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy import interpolate
import matplotlib.cm as cm
import os
from dataloader.SpectrumObject import SpectrumObject


def spectra_comparison(spectra_list, title="Spectra Comparison"):
    """
    Plots multiple spectra on the same graph for visual comparison.

    Parameters:
    - spectra_list (list of tuples): Each tuple should be (mz, intensity, metadata_label).
    - title (str, optional): Title of the plot.
    
    Example of `spectra_list` input:
        [(mz1, intensity1, "Escherichia-Coli-DRIAMS_A-2015"),
         (mz2, intensity2, "Klebsiella-Pneumoniae-DRIAMS_B-2016")]
    """
    if len(spectra_list) == 0:
        print("⚠️ No spectra to compare.")
        return

    plt.figure()

    # Generate random colors for each spectrum
    random.seed(42)  # Ensure consistent colors across runs
    colors = [plt.cm.viridis(i / len(spectra_list)) for i in range(len(spectra_list))]

    # Plot each spectrum
    for i, (spectrum, label) in enumerate(spectra_list):
        mz, intensity = spectrum.mz, spectrum.intensity
        plt.plot(mz, intensity, label=label, color=colors[i], linewidth=1.5)

    # Formatting
    plt.xlabel("m/z")
    plt.ylabel("Intensity")
    plt.title(title)
    plt.legend(loc="upper right", fontsize=8)  # Show legend with metadata labels
    plt.grid(alpha=0.3)
    plt.show()

def interpolate_spectrum(spectrum, target_length=1000):
    """Interpolates a spectrum to a fixed length."""
    if len(spectrum[0]) < 2:  # Ensure spectrum has at least two points
        raise ValueError("Spectrum must have at least two points for interpolation.")

    # Create interpolation function
    f = interpolate.interp1d(spectrum[0], spectrum[1], kind='linear', fill_value="extrapolate")

    # Generate new m/z values
    mz_new = np.linspace(spectrum[0].min(), spectrum[0].max(), target_length)
    intensity_new = f(mz_new)

    return intensity_new  # Only return interpolated intensities

def plot_pca(dataset, target="bacteria", target_length=1000, n_components=2, title = None):
    """
    Performs PCA on multiple datasets and plots the first two principal components.

    Parameters:
    - dataset (DRIAMS_Dataset): Dataset to visualize.
    - target (str, optional): What to group spectra by (options: "bacteria", "hospital", "year").
    - target_length (int, optional): Fixed length for interpolation before PCA.
    - n_components (int, optional): Number of PCA components to use.

    Returns:
    - None (displays a scatter plot)
    """

    title = f"PCA Projection of Spectra by {target.capitalize()}" if title is None else title

    X, y = [], []

    for i in range(len(dataset)):
        spectrum, metadata = dataset[i]  # Retrieve spectrum data
        mz, intensity = spectrum.mz, spectrum.intensity
        interpolated_intensity = interpolate_spectrum((mz, intensity), target_length)

        # Skip NaN-containing spectra
        if np.isnan(interpolated_intensity).any():
            continue

        X.append(interpolated_intensity)

        # Extract label based on the target grouping (bacteria, hospital, or year)
        genus, species, hospital, year = metadata.split("-")
        if target == "bacteria":
            y.append(f"{genus} {species}")
        elif target == "hospital":
            y.append(hospital)
        elif target == "year":
            y.append(year)
        else:
            raise ValueError("Invalid target. Choose 'bacteria', 'hospital', or 'year'.")

    X = np.array(X)

    if len(X) == 0:
        raise ValueError("No valid spectra available for PCA after removing NaN values.")

    # Perform PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)

    # Create color map
    unique_labels = list(set(y))
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))

    # Plot PCA result
    plt.figure(figsize=(6, 4))
    for i, label in enumerate(unique_labels):
        mask = np.array(y) == label
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1], alpha=0.6, color=colors[i], label=label)

    # Labels and Title
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)

    # Show plot
    plt.show()

def plot_tsne(dataset, target="bacteria", target_length=1000, perplexity=30, learning_rate=200, title=None):
    """
    Plots a t-SNE visualization for one or multiple datasets in a single plot.

    Parameters:
    - datasets (DRIAMS_Dataset): Dataset to visualize.
    - target (str, optional): What to group spectra by (options: "bacteria", "hospital", "year").
    - target_length (int, optional): Fixed length for interpolation before t-SNE.
    - perplexity (int, optional): Perplexity parameter for t-SNE.
    - learning_rate (int, optional): Learning rate parameter for t-SNE.
    - title (str, optional): Custom title for the plot (default is auto-generated).

    Returns:
    - None (displays a scatter plot)
    """

    # Auto-generate title if none provided
    if title is None:
        title = f"t-SNE Projection of Spectra by {target.capitalize()}"

    X, y = [], []


    for i in range(len(dataset)):
        spectrum, metadata = dataset[i]  # Retrieve spectrum data
        mz, intensity = spectrum.mz, spectrum.intensity
        interpolated_intensity = interpolate_spectrum((mz, intensity), target_length)

        # Skip NaN-containing spectra
        if np.isnan(interpolated_intensity).any():
            continue

        X.append(interpolated_intensity)

        # Extract label based on the target grouping
        genus, species, hospital, year = metadata.split("-")
        if target == "bacteria":
            y.append(f"{genus} {species}")  # Example: "Escherichia Coli"
        elif target == "hospital":
            y.append(hospital)  # Example: "DRIAMS_A"
        elif target == "year":
            y.append(year)  # Example: "2018"
        else:
            raise ValueError("Invalid target. Choose 'bacteria', 'hospital', or 'year'.")

    X = np.array(X)

    if len(X) == 0:
        raise ValueError("No valid spectra available for t-SNE after removing NaN values.")

    # Perform t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate=learning_rate, random_state=42, n_jobs=-1)
    X_tsne = tsne.fit_transform(X)

    # Create color map
    unique_labels = list(set(y))
    color_map = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))

    # Plot t-SNE result
    plt.figure(figsize=(6, 4))
    for i, label in enumerate(unique_labels):
        mask = np.array(y) == label
        plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1], alpha=0.6, color=color_map[i], label=label)

    # Labels and Title
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)

    # Show plot
    plt.show()

def visualize_preprocessing_steps(spectrum, pipeline):
    """
    Visualizes a random spectrum at each step of the preprocessing pipeline.

    Parameters:
    - spectrum (SpectrumObject): The original spectrum.
    - pipeline (SequentialPreprocessor): The preprocessing pipeline.
    
    Returns:
    - None (Displays plots)
    """
    plt.figure(figsize=(10, 6))

    # Start with original spectrum
    spectrum, metadata = spectrum
    mz, intensity = spectrum.mz, spectrum.intensity
    plt.plot(mz, intensity, label=metadata, linestyle="dashed", alpha=0.8)

    # Apply preprocessing step by step
    for step in pipeline.preprocessors:
        spectrum = step(spectrum)  # Apply step
        mz, intensity = spectrum.mz, spectrum.intensity  # Extract new values
        plt.plot(mz, intensity, label=f"After {step.__class__.__name__}")

    # Formatting
    plt.xlabel("m/z")
    plt.ylabel("Intensity")
    plt.title("Preprocessing Step-by-Step on a Random Sample")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

#### MLP_VAE ####

def plot_tsne(bot, z, labels=None, perplexity=30, random_state=42, path='.', name='vae', epoch=0):
    """
    Plots and saves t-SNE of both the MLP bottleneck and VAE latent space.

    Parameters:
    - bot (torch.Tensor): Output of encoder_bot [N, D].
    - z (torch.Tensor): Latent vectors after reparameterization [N, d].
    - labels (list of str): Labels for color grouping (optional).
    - perplexity (int): t-SNE perplexity parameter.
    - random_state (int): t-SNE random state.
    - path (str): Folder to save the plot.
    - name (str): Prefix name for the saved file.
    - epoch (int): Epoch number for file naming.
    """
    assert bot.shape[0] == z.shape[0], "bot and z must have the same number of samples"

    bot_np = bot.cpu().numpy()
    z_np = z.cpu().numpy()

    tsne_bot = TSNE(n_components=2, perplexity=perplexity, random_state=random_state)
    tsne_z = TSNE(n_components=2, perplexity=perplexity, random_state=random_state)

    bot_2d = tsne_bot.fit_transform(bot_np)
    z_2d = tsne_z.fit_transform(z_np)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    if labels is not None:
        labels = np.array(labels)
        for ax, data, title in zip(axes, [bot_2d, z_2d], ["MLP Bottleneck", "VAE Latent z"]):
            for label in np.unique(labels):
                idx = labels == label
                ax.scatter(data[idx, 0], data[idx, 1], label=str(label), s=10)
            ax.set_title(f"t-SNE of {title}")
            ax.legend(fontsize=6)
    else:
        axes[0].scatter(bot_2d[:, 0], bot_2d[:, 1], s=10)
        axes[0].set_title("t-SNE of MLP Bottleneck")
        axes[1].scatter(z_2d[:, 0], z_2d[:, 1], s=10)
        axes[1].set_title("t-SNE of VAE Latent z")

    for ax in axes:
        ax.set_xlabel("t-SNE-1")
        ax.set_ylabel("t-SNE-2")

    plt.tight_layout()
    filename = os.path.join(path, f"{name}_tsne_epoch_{epoch}.png")
    plt.savefig(filename)
    plt.close()
    print(f"✅ t-SNE plot saved to {filename}")

def visualize_preprocessing(sample, pipeline, path):
    """
    Visualizes a random spectrum at each step of the preprocessing pipeline,
    each step in a different subplot.

    Parameters:
    - spectrum (SpectrumObject): The original spectrum.
    - pipeline (SequentialPreprocessor): The preprocessing pipeline.
    
    Returns:
    - None (Displays plots)
    """

    spectrum, metadata = sample

    steps = [("Raw", spectrum)]

    # Apply preprocessing step by step and save intermediate results
    for step in pipeline.preprocessors:
        spectrum = step(spectrum)
        steps.append((step.__class__.__name__, spectrum))

    # Plot each step in its own subplot
    num_steps = len(steps)
    fig, axes = plt.subplots(num_steps, 1, figsize=(10, 3 * num_steps), sharex=True)

    if num_steps == 1:
        axes = [axes]  # Make iterable if only one subplot

    for ax, (name, spec) in zip(axes, steps):
        mz, intensity = spec.mz, spec.intensity
        ax.plot(mz, intensity)
        ax.set_title(f"{name} Spectrum")
        ax.set_ylabel("Intensity")
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel("m/z")
    fig.suptitle(f"Preprocessing Steps for {metadata}", fontsize=14, y=1.02)
    plt.tight_layout()

    metadata = metadata.replace("/", "_")
    filename = os.path.join(path, f"preproc_{metadata}.png")
    plt.savefig(filename)
    plt.close()
    print(f"✅ Preprocessing plot saved to {filename}")