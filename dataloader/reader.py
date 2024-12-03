import numpy as np
import pandas as pd
from scipy.stats import binned_statistic
import os
from scipy.signal import savgol_filter
from scipy import sparse
from scipy.linalg import norm
import pandas as pd
import numpy as np
from scipy.stats import binned_statistic
import matplotlib.pyplot as plt
from dataloader.preprocess import SequentialPreprocessor, SpectrumObject, VarStabilizer, Smoother, BaselineCorrecter, Trimmer, Binner, Normalizer



# Code copied from maldi-nn, thanks to the authors. Adapted by Alejandro Guerrero-López.

   
class MaldiDataset:
    def __init__(self, root_dir, n_step=3):
        self.root_dir = root_dir
        self.n_step = n_step
        self.data = []

    def parse_dataset(self):
        for year_folder in os.listdir(self.root_dir):
            year_folder_path = os.path.join(self.root_dir, year_folder)
            if os.path.isdir(year_folder_path):
                # Extract the year label
                year_label = year_folder

                # Iterate through genus folders
                for genus_folder in os.listdir(year_folder_path):
                    genus_folder_path = os.path.join(year_folder_path, genus_folder)
                    if os.path.isdir(genus_folder_path):
                        # Extract genus label
                        genus_label = genus_folder

                        # Iterate through species folders
                        for species_folder in os.listdir(genus_folder_path):
                            species_folder_path = os.path.join(genus_folder_path, species_folder)
                            if os.path.isdir(species_folder_path):
                                # Extract genus+species label
                                genus_species_label = f"{genus_label} {species_folder}"

                                # Iterate through replicate folders
                                for replicate_folder in os.listdir(species_folder_path):
                                    replicate_folder_path = os.path.join(species_folder_path, replicate_folder)
                                    if os.path.isdir(replicate_folder_path):
                                        # Iterate through lecture folders
                                        for lecture_folder in os.listdir(replicate_folder_path):
                                            lecture_folder_path = os.path.join(replicate_folder_path, lecture_folder)
                                            if os.path.isdir(lecture_folder_path):
                                                # Search for "acqu" and "fid" files
                                                acqu_file, fid_file = self._find_acqu_fid_files(lecture_folder_path)
                                                if acqu_file and fid_file:
                                                    # Read the maldi-tof spectra using from_bruker
                                                    spectrum = SpectrumObject.from_bruker(acqu_file, fid_file)
                                                    # Binarize the spectrum using Binner
                                                    binner = SequentialPreprocessor(
                                                        VarStabilizer(method="sqrt"),
                                                        Smoother(halfwindow=10),
                                                        BaselineCorrecter(method="SNIP", snip_n_iter=20),
                                                        Trimmer(),
                                                        Binner(step=self.n_step),
                                                        Normalizer(sum=1),
                                                    )
                                                    binned_spectrum = binner(spectrum)
                                                    # Append data point to the dataset
                                                    # Skip if the spectrum is NaN due to preprocessing
                                                    if np.isnan(binned_spectrum.intensity).any():
                                                        print("Skipping NaN spectrum")
                                                        continue
                                                    self.data.append({
                                                        'spectrum': binned_spectrum.intensity,
                                                        'm/z': binned_spectrum.mz,
                                                        'year_label': year_label,
                                                        'genus_label': genus_label,
                                                        'genus_species_label': genus_species_label,
                                                    })

    def _parse_folder_name(self, folder_name):
        # Split folder name into genus, species, and hospital code
        parts = folder_name.split()
        genus_species = " ".join(parts[:2])
        hospital_code = " ".join(parts[2:])
        return genus_species, hospital_code

    def _find_acqu_fid_files(self, directory):
        acqu_file = None
        fid_file = None
        for root, _, files in os.walk(directory):
            for file in files:
                if file == 'acqu':
                    acqu_file = os.path.join(root, file)
                elif file == 'fid':
                    fid_file = os.path.join(root, file)
                if acqu_file and fid_file:
                    return acqu_file, fid_file
        return acqu_file, fid_file

    def get_data(self):
        return self.data

