import os
import sys
import torch
import time
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dataloader.MaldiMaranon_Manager import MaldiMaranonManager
from dataloader.MaldiDataset import MaldiDataset, SynthDataset
from utils.preprocess import SequentialPreprocessor, VarStabilizer, Smoother, BaselineCorrecter, Trimmer, Binner, Normalizer, StdThresholder, MinMaxScaler, LogScaler
from models.bottlenecks import MLP
from models.AE_VAE import VAE
from utils.misc import plot_train_val_curves, early_stopping, train, evaluate, collate_spectra, predict, test_synth_data

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def main():

    # ------------------------------
    # 3) TRAIN
    # ------------------------------

    nll_curve_train = []
    nll_curve_val = []
    RE_curve_train = []
    RE_curve_val = []
    KL_curve_train = []
    KL_curve_val = []

    epochs = 200
    learning_rate = 1e-3
    max_patience = 10
    patience = 0
    best_nll = float('inf')
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

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

def inference(model, test_loader, lastpreprocessing, device, path, name):

    return predict(model, test_loader, lastpreprocessing, device, path, name, save_synth=True)

if __name__ == "__main__":

    training = False

    saved = '/export/usuarios_ml4ds/lschmidt/GITHUB/MALDI-UC3M/results/MALDIS_mlp_vae_20250526_154505'

    data_name = 'MALDIS'
    preproc = 'log10'
    loss_mode = 'mse'
    name = 'mlp_vae'
    result_dir = f'/export/usuarios_ml4ds/lschmidt/GITHUB/MALDI-UC3M/results/{data_name}_{name}_{time.strftime("%Y%m%d_%H%M%S")}/' if training else saved
    name = f'{name}_{preproc}_{loss_mode}'
    os.makedirs(result_dir, exist_ok=True)


    # ------------------------------
    # 1) SETUP: hyperparams
    # ------------------------------

    dataset_path = f"/export/data_ml4ds/bacteria_id/MaldiMaranonDB"

    if preproc == 'log10':
        last_preprocess = LogScaler(base=10)
        last_activation = 'relu'
    elif preproc == 'minmax':
        last_preprocess = MinMaxScaler()
        last_activation = 'sigmoid'
    elif preproc == 'norm':
        last_preprocess = Normalizer(sum=1)
        last_activation = 'sigmoid'
    else:
        raise ValueError(f"Unknown preprocessing method: {preproc}")

    preprocess_pipeline = SequentialPreprocessor(VarStabilizer(method="sqrt"),
                                                Smoother(halfwindow=10),
                                                BaselineCorrecter(method="SNIP", snip_n_iter=20),
                                                StdThresholder(factor=1.0),
                                                Trimmer(),
                                                Binner(step=3),
                                                last_preprocess)                                        

    # Initialize the DRIAMS manager
    pickle_path = os.path.join(os.path.dirname(__file__), 'maldi_manager.pkl')
    manager = MaldiMaranonManager(dataset_path, presaved=True, pickel_path=pickle_path)

    training_years = ['2022', '2021', '2020', '2019', '2018']
    train_data = manager.query_spectra_dict(years=training_years, genus='Escherichia', species='Coli')
    val_data = manager.query_spectra_dict(years='2023', genus='Escherichia', species='Coli')
    test_data = manager.query_spectra_dict(years='2024',  genus='Escherichia', species='Coli')

    # Create datasets
    train_dataset = MaldiDataset(train_data, preprocess_pipeline=preprocess_pipeline, visualize=training, path=result_dir)
    val_dataset   = MaldiDataset(val_data, preprocess_pipeline=preprocess_pipeline)  
    test_dataset  = MaldiDataset(test_data, preprocess_pipeline=preprocess_pipeline)

    # DataLoader for training, validation, and test sets.
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_spectra)
    val_loader   = DataLoader(val_dataset,   batch_size=64, shuffle=False, collate_fn=collate_spectra)
    test_loader  = DataLoader(test_dataset,  batch_size=64, shuffle=False, collate_fn=collate_spectra)


    # ------------------------------
    # 2) DEFINE MODELS
    # ------------------------------

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define the encoder and decoder networks.
    bottleneck = MLP(input_dim=6000, last=last_activation) # This can be replaced with any other bottleneck architecture.
    encoder_bot = bottleneck.encoder
    decoder_bot = bottleneck.decoder
    model = VAE(encoder_bot, decoder_bot, loss_mode=loss_mode).to(device)

    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    saved_model = main() if training else f'{saved}/mlp_vae_bestmodel.pth'
    model.load_state_dict(torch.load(saved_model, map_location=device))

    # ------------------------------
    # 5) INFERENCE & SYNTHESIS
    # ------------------------------

    path = path = os.path.dirname(saved_model)
    name = f'Inference_{preproc}_{loss_mode}'
    synth_data = inference(model, test_loader, preproc, device, path, name)    
    synth_dataset = SynthDataset(synth_data)
    
    # Test the synthesized dataset
    rf_model_path = '/export/usuarios_ml4ds/lschmidt/GITHUB/MALDI-UC3M/results/MALDIS_rf_20250515_133615/rf_stdthr_model.pkl'
    test_synth_data(synth_dataset, rf_model_path)
