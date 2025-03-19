import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy import interpolate
import matplotlib.cm as cm

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
    for i, (mz, intensity, label) in enumerate(spectra_list):
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
    - datasets (list): List of DRIAMS_Dataset instances.
    - target (str, optional): What to group spectra by (options: "bacteria", "hospital", "year").
    - target_length (int, optional): Fixed length for interpolation before PCA.
    - n_components (int, optional): Number of PCA components to use.

    Returns:
    - None (displays a scatter plot)
    """

    title = f"PCA Projection of Spectra by {target.capitalize()}" if title is None else title

    X, y = [], []

    for i in range(len(dataset)):
        mz, intensity, metadata = dataset[i]  # Retrieve spectrum data
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
    - datasets (list or DRIAMS_Dataset): Dataset(s) to visualize.
    - target (str, optional): What to group spectra by (options: "bacteria", "hospital", "year").
    - target_length (int, optional): Fixed length for interpolation before t-SNE.
    - perplexity (int, optional): Perplexity parameter for t-SNE.
    - learning_rate (int, optional): Learning rate parameter for t-SNE.
    - title (str, optional): Custom title for the plot (default is auto-generated).

    Returns:
    - None (displays a scatter plot)
    """

    if not isinstance(dataset, list):  # If a single dataset is given, wrap it in a list
        dataset = [dataset]

    # Auto-generate title if none provided
    if title is None:
        title = f"t-SNE Projection of Spectra by {target.capitalize()}"

    X, y = [], []


    for i in range(len(dataset)):
        mz, intensity, metadata = dataset[i]  # Retrieve spectrum data
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
    tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate=learning_rate, random_state=42)
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