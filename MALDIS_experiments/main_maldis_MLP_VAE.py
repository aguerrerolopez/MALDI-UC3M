import os
import sys
import torch
import time
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dataloader.MaldiMaranon_Manager import MaldiMaranonManager
from dataloader.MaldiDataset import MaldiDataset, SynthDataset
from utils.preprocess import SequentialPreprocessor, VarStabilizer, Smoother, BaselineCorrecter, Trimmer, Binner, Normalizer, StdThresholder, MinMaxScaler
from models.bottlenecks import MLP
from models.AE_VAE import VAE
from utils.misc import plot_train_val_curves, early_stopping, train, evaluate, collate_spectra, predict, test_synth_data

def main():

    # ------------------------------
    # 1) SETUP: data, hyperparams
    # ------------------------------

    data_name = 'MALDIS'
    name = 'mlp_vae'
    result_dir = f'/export/usuarios_ml4ds/lschmidt/GITHUB/MALDI-UC3M/results/{data_name}_{name}_{time.strftime("%Y%m%d_%H%M%S")}/'
    os.makedirs(result_dir, exist_ok=True)

    # Set device and hyperparameters.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 200
    learning_rate = 1e-3
    loss_mode = 'mse'  # Change to 'mse' or 'gaussian' if desired.

    # MALDIMARANON dataset
    # Load full training dataset
    dataset_path = f"/export/data_ml4ds/bacteria_id/MaldiMaranonDB"

    binning_step = 3
    preprocess_pipeline = SequentialPreprocessor(VarStabilizer(method="sqrt"),
                                                Smoother(halfwindow=10),
                                                BaselineCorrecter(method="SNIP", snip_n_iter=20),
                                                StdThresholder(factor=1.0),
                                                Trimmer(),
                                                Binner(step=binning_step),
                                                MinMaxScaler())
                                                # Normalizer(sum=1))

    # Initialize the DRIAMS manager
    pickle_path = os.path.join(os.path.dirname(__file__), 'maldi_manager.pkl')
    manager = MaldiMaranonManager(dataset_path, presaved=True, pickel_path=pickle_path)

    val_data = manager.query_spectra_dict(years='2023', genus='Escherichia', species='Coli')
    training_years = ['2022', '2021', '2020', '2019', '2018']
    train_data = manager.query_spectra_dict(years=training_years, genus='Escherichia', species='Coli')

    # Create datasets
    train_dataset = MaldiDataset(train_data, preprocess_pipeline=preprocess_pipeline, visualize=True, path=result_dir)
    val_dataset   = MaldiDataset(val_data, preprocess_pipeline=preprocess_pipeline)  

    # DataLoader for training, validation, and test sets.
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_spectra)
    val_loader   = DataLoader(val_dataset,   batch_size=64, shuffle=False, collate_fn=collate_spectra)

    # ------------------------------
    # 2) DEFINE MODELS
    # ------------------------------

    # Define the encoder and decoder networks.
    bottleneck = MLP(input_dim=6000) # This can be replaced with any other bottleneck architecture.
    encoder_bot = bottleneck.encoder
    decoder_bot = bottleneck.decoder

    # Instantiate the model, optimizer.
    model = VAE(encoder_bot, decoder_bot, loss_mode=loss_mode).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # ------------------------------
    # 3) TRAIN
    # ------------------------------

    nll_curve_train = []
    nll_curve_val = []
    RE_curve_train = []
    RE_curve_val = []
    KL_curve_train = []
    KL_curve_val = []

    max_patience = 10
    patience = 0
    best_nll = float('inf')

    for epoch in range(1, epochs + 1):
        nll, re, kl = train(model, device, train_loader, optimizer, epoch)
        nll_curve_train.append(nll)
        RE_curve_train.append(re)
        KL_curve_train.append(kl)

        # ------------------------------
        # 4) VALIDATE
        # ------------------------------

        nll_val, re_val, kl_val = evaluate(val_loader, model=model, epoch=epoch, device=device)
        nll_curve_val.append(nll_val)
        RE_curve_val.append(re_val)
        KL_curve_val.append(kl_val)

        # Early stopping check and save best model
        early_stopped, best_nll, patience, saved_path = early_stopping(epoch, nll_val, best_nll, patience, max_patience, model, name, result_dir, saving='best')

        if early_stopped:
            print(f"Early stopping at epoch {epoch} with a loss of {best_nll}.")
            print(f"Best model saved at: {saved_path}")
            break

    train_data = [nll_curve_train, RE_curve_train, KL_curve_train]
    val_data = [nll_curve_val, RE_curve_val, KL_curve_val]

    plot_train_val_curves(result_dir + name, train_data, val_data)

    return saved_path

def inference(model, test_loader, device, path, name):

    return predict(model, test_loader, device, path, name, save_synth=True)

if __name__ == "__main__":

    training = True

    # MALDIMARANON dataset
    dataset_path = f"/export/data_ml4ds/bacteria_id/MaldiMaranonDB"

    binning_step = 3
    preprocess_pipeline = SequentialPreprocessor(VarStabilizer(method="sqrt"),
                                                Smoother(halfwindow=10),
                                                BaselineCorrecter(method="SNIP", snip_n_iter=20),
                                                StdThresholder(factor=1.0),
                                                Trimmer(),
                                                Binner(step=binning_step),
                                                MinMaxScaler())
                                                # Normalizer(sum=1))

    # Initialize the DRIAMS manager
    pickle_path = os.path.join(os.path.dirname(__file__), 'maldi_manager.pkl')
    manager = MaldiMaranonManager(dataset_path, presaved=True, pickel_path=pickle_path)
    test_data = manager.query_spectra_dict(years='2024',  genus='Escherichia', species='Coli')
    test_dataset  = MaldiDataset(test_data, preprocess_pipeline=preprocess_pipeline)
    test_loader  = DataLoader(test_dataset,  batch_size=64, shuffle=False, collate_fn=collate_spectra)

    print(f"Length of the test dataset: {len(test_dataset)}")
    
    # ------------------------------
    # 2) DEFINE MODELS
    # ------------------------------

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_mode = 'mse'

    bottleneck = MLP(input_dim=6000)
    encoder_bot = bottleneck.encoder
    decoder_bot = bottleneck.decoder
    model = VAE(encoder_bot, decoder_bot, loss_mode=loss_mode).to(device)


    saved_model = main() if training else '/export/usuarios_ml4ds/lschmidt/GITHUB/MALDI-UC3M/results/MALDIS_mlp_vae_20250514_172756/mlp_vae_bestmodel.pth'
    model.load_state_dict(torch.load(saved_model))

    path = path = os.path.dirname(saved_model)
    name = 'Inference_mlp_vae'
    synth_data = inference(model, test_loader, device, path, name)    

    synth_dataset = SynthDataset(synth_data)
    print(f"Length of the synthesized dataset: {len(synth_dataset)}")
    
    # Test the synthesized dataset
    # rf_model_path = '/export/usuarios_ml4ds/lschmidt/GITHUB/MALDI-UC3M/results/MALDIS_rf_20250514_131530/rf_stdthr_model.pkl'
    # test_synth_data(synth_dataset, rf_model_path)


