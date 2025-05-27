import os
import sys
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.losses import loss_function
from utils.visualization import plot_tsne, plot_samples, get_mean_spectra, plot_pca_2d, plot_pca_3d, plot_umap
from dataloader.SpectrumObject import SpectrumObject


def collate_spectra(batch):
    intensities = torch.stack([torch.tensor(sample[0].intensity, dtype=torch.float32) for sample in batch])
    mzs = torch.stack([torch.tensor(sample[0].mz, dtype=torch.float32) for sample in batch])
    labels = [sample[1] for sample in batch]
    metadata = [sample[2] for sample in batch]
    return (intensities, mzs), labels, metadata

def plot_train_val_curves(name, train_data, val_data, title="_NLL_RE_KL", legend=['NLL', 'RE', 'KL']):

    assert len(train_data) == len(val_data) == 3, "train_data and val_data must contain as many elements as legend values (default: NLL, RE, KL)."

    nll_curve_train, RE_curve_train, KL_curve_train = train_data
    nll_curve_val, RE_curve_val, KL_curve_val = val_data

    plt.figure(figsize=(10, 6))

    # Colors: NLL = blue, RE = orange, KL = green
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    epochs = np.arange(len(nll_curve_train))

    # Plot training curves
    plt.plot(epochs, nll_curve_train, linestyle='-', color=colors[0], label=legend[0] + ' (train)')
    plt.plot(epochs, RE_curve_train, linestyle='-', color=colors[1], label=legend[1] + ' (train)')
    plt.plot(epochs, KL_curve_train, linestyle='-', color=colors[2], label=legend[2] + ' (train)')

    # Plot validation curves
    plt.plot(epochs, nll_curve_val, linestyle='--', color=colors[0], label=legend[0] + ' (val)')
    plt.plot(epochs, RE_curve_val, linestyle='--', color=colors[1], label=legend[1] + ' (val)')
    plt.plot(epochs, KL_curve_val, linestyle='--', color=colors[2], label=legend[2] + ' (val)')

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(name + title + ".pdf", bbox_inches='tight')
    plt.close()

def early_stopping(epoch, nll_val, best_nll, patience, max_patience, model, name, path, saving='best'):
    """
    Early stopping function
    :param epoch: current epoch
    :param nll_val: negative log-likelihood values
    :param best_nll: best negative log-likelihood value
    :param patience: current patience
    :param max_patience: maximum patience
    :return: updated patience and best_nll
    """
    saved_path = os.path.join(path, f"{name}_bestmodel.pth") if saving == 'best' else os.path.join(path, f"{name}_epoch_{epoch}.pth")
    
    if epoch == 0 or (nll_val < best_nll):
        torch.save(model.state_dict(), saved_path)
        print("saved!")
        best_nll = nll_val
        patience = 0
    else:
        patience += 1

    if patience > max_patience:
        return True, best_nll, patience, saved_path

    return False, best_nll, patience, saved_path

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0.0
    RE_vals = 0.0
    KL_vals = 0.0

    for batch_idx, batch in enumerate(train_loader):
        spectra, labels, metas = batch
        intensity, mz = spectra

        data = intensity.to(device)

        optimizer.zero_grad()
        recon_batch, mu, logvar, _, _ = model(data)

        loss, rec, kl = loss_function(recon_batch, data, mu, logvar, model.loss_mode)
        loss.backward()

        optimizer.step()

        train_loss += loss.item()
        RE_vals += rec.item()
        KL_vals += kl.item()
        
        if batch_idx % 100 == 0:
            print(f"Epoch {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] Loss: {loss.item() / len(data):.4f}  RECON: {rec.item() / len(data):.4f}  KL: {kl.item() / len(data):.4f}", flush=True)
    print(f"====> Epoch {epoch} Average loss: {train_loss / len(train_loader.dataset):.4f}  RECON: {RE_vals / len(train_loader.dataset):.4f}  KL: {KL_vals / len(train_loader.dataset):.4f}")

    return train_loss / len(train_loader.dataset), RE_vals / len(train_loader.dataset), KL_vals / len(train_loader.dataset)

def evaluate(test_loader, name=None, model=None, epoch=None, device="cpu"):

    if model is None:
        assert name is not None, "If no model is passed, 'name' must be given to load the model."
        model = torch.load(name + '.model', map_location=device)

    model.to(device)
    model.eval()

    total_loss = 0.0
    total_RE = 0.0
    total_KL = 0.0

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            spectra, labels, metas = batch
            intensity, _ = spectra

            data = intensity.to(device)

            recon_batch, mu, logvar, _, _ = model(data)
            loss, rec, kl = loss_function(recon_batch, data, mu, logvar, model.loss_mode)
            
            total_loss += loss.item()
            total_RE += rec.item()
            total_KL += kl.item()

    avg_nll = total_loss / len(test_loader.dataset)
    avg_RE = total_RE / len(test_loader.dataset)
    avg_KL = total_KL / len(test_loader.dataset)

    if epoch is not None:
        print(f"Epoch {epoch} VALIDATION: NLL: {avg_nll:.4f} | RE: {avg_RE:.4f} | KL: {avg_KL:.4f}")
    else:
        print(f"FINAL VALIDATION: NLL: {avg_nll:.4f} | RE: {avg_RE:.4f} | KL: {avg_KL:.4f}")

    return avg_nll, avg_RE, avg_KL

def predict(model, test_loader, lastpreprocessing, device, result_dir, name, num_samples_to_plot=5, save_synth=False):
    model.to(device)
    model.eval()

    os.makedirs(result_dir, exist_ok=True)
    plotted = 0
    selected_info = {}
    selected_indices = random.sample(range(len(test_loader.dataset)), min(num_samples_to_plot, len(test_loader.dataset)))

    samples = []
    original= []
    reconstructed = []
    global_indices = []
    synth_data = [] if save_synth else None

    all_z = []

    total_loss = 0.0
    total_RE = 0.0
    total_KL = 0.0

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            spectra, labels, metas = batch
            intensities, mzs = spectra
            intensities = intensities.to(device)

            recon_batch, mu, logvar, bot, z = model(intensities)
            all_z.append(z.cpu().numpy())
            loss, rec, kl = loss_function(recon_batch, intensities, mu, logvar, model.loss_mode)

            for i in range(intensities.size(0)):
                int = intensities[i].cpu().numpy()
                recon = recon_batch[i].cpu().numpy()

                if lastpreprocessing == 'log10':
                    # Inverse log10 scaling
                    int = 10 ** int - 1
                    recon = 10 ** recon - 1

                mz = mzs[i].cpu().numpy()
                label = labels[i] + '_synth'
                meta = metas[i]

                global_idx = batch_idx * test_loader.batch_size + i

                if save_synth:
                    synthetic_spectrum = SpectrumObject(mz, recon)
                    synth_data.append((synthetic_spectrum, label, meta))


                # Plot only the selected samples
                if global_idx in selected_indices and plotted < num_samples_to_plot:
                    samples.append((int, recon, global_idx))
                    global_indices.append(global_idx)
                    selected_info[global_idx] = meta['study']
                    plotted += 1

                original.append(int)
                reconstructed.append(recon)

            total_loss += loss.item()
            total_RE += rec.item()
            total_KL += kl.item()

        all_z = np.concatenate(all_z, axis=0)

        print(f"====>TEST: Average loss: {total_loss / len(test_loader.dataset):.4f}  RECON: {total_RE / len(test_loader.dataset):.4f}  KL: {total_KL / len(test_loader.dataset):.4f}")
    
    # Plotting
    plot_samples(samples, result_dir, name, labels=selected_info)
    plot_tsne(all_z, original, global_indices, result_dir, name)
    plot_pca_2d(all_z, original, global_indices, result_dir, name)
    plot_pca_3d(all_z, original, global_indices, result_dir, name)
    plot_umap(all_z, original, global_indices, result_dir, name)
    get_mean_spectra([original, reconstructed], ['Original', 'Reconstructed'], result_dir, name)

    return synth_data if save_synth else None


def test_synth_data(synth_dataset, rf_model_path):
    """
    Test a pre-trained Random Forest model on the synthetic dataset.
    Args:
        synth_dataset: Dataset containing (SpectrumObject, label, meta) tuples, where label ends with '_synth'.
        rf_model_path: Path to the saved Random Forest model (torch.save'd .pth file).
    Returns:
        acc: Accuracy score.
        report: Full classification report (as string).
    """

    results_dir = os.path.dirname(rf_model_path)

    # Load trained RF model
    rf = joblib.load(rf_model_path)

    # Prepare test data
    X_test = []
    y_true = []

    for spectrum, label, _ in synth_dataset:
        X_test.append(spectrum.intensity)
        # Remove "_synth" from label to compare with RF trained on true labels
        y_true.append(label.replace('_synth', ''))

    # Convert to numpy
    X_test = np.stack(X_test)

    # Predict
    y_pred = rf.predict(X_test)

    # Evaluate
    acc = accuracy_score(y_true, y_pred)
    print(f"RF Accuracy on synthetic data: {acc:.4f}")

    report = classification_report(y_true, y_pred)
    print("Classification Report:\n", report)
    with open(os.path.join(results_dir, "classification_report.txt"), "w") as f:
        f.write(report)

    _, ax2 = plt.subplots(figsize=(8, 6))
    cnf = confusion_matrix(y_true, y_pred, labels=rf.classes_)
    ConfusionMatrixDisplay(cnf, display_labels=rf.classes_).plot(ax=ax2, xticks_rotation=45)
    plt.title("Confusion Matrix - Synthetic Data")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "confusion_matrix_synth.png"))
    print("Confusion Matrix:\n", confusion_matrix)

    return