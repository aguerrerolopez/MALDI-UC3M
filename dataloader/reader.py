import numpy as np
import pandas as pd
import os
import pandas as pd
import numpy as np
from dataloader.preprocess import SequentialPreprocessor, SpectrumObject, VarStabilizer, Smoother, BaselineCorrecter, Trimmer, Binner, Normalizer
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


# Code copied from maldi-nn, thanks to the authors. Adapted by Alejandro Guerrero-López.

   
class MaldiDataset:
    def __init__(self, root_dir, n_step=3):
        self.root_dir = root_dir
        self.n_step = n_step
        self.data = []

    def parse_dataset(self):
        # Initialize a ThreadPoolExecutor for parallel processing
        max_workers = os.cpu_count() or 4  # Use number of CPUs or default to 4
        futures = []

        def process_replicate_folder(replicate_folder_path, genus_label, genus_species_label, year_label):
            data_points = []
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
                        # Skip if the spectrum is NaN due to preprocessing
                        if np.isnan(binned_spectrum.intensity).any():
                            print("Skipping NaN spectrum")
                            continue
                        data_points.append({
                            'spectrum': binned_spectrum.intensity,
                            'm/z': binned_spectrum.mz,
                            'year_label': year_label,
                            'genus_label': genus_label,
                            'genus_species_label': genus_species_label,
                        })
            return data_points

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for year_folder in os.listdir(self.root_dir):
                year_folder_path = os.path.join(self.root_dir, year_folder)
                if os.path.isdir(year_folder_path):
                    year_label = year_folder
                    for genus_folder in os.listdir(year_folder_path):
                        genus_folder_path = os.path.join(year_folder_path, genus_folder)
                        if os.path.isdir(genus_folder_path):
                            genus_label = genus_folder
                            for species_folder in os.listdir(genus_folder_path):
                                species_folder_path = os.path.join(genus_folder_path, species_folder)
                                if os.path.isdir(species_folder_path):
                                    genus_species_label = f"{genus_label} {species_folder}"
                                    for replicate_folder in os.listdir(species_folder_path):
                                        replicate_folder_path = os.path.join(species_folder_path, replicate_folder)
                                        if os.path.isdir(replicate_folder_path):
                                            # Submit replicate folder processing to executor
                                            futures.append(
                                                executor.submit(
                                                    process_replicate_folder,
                                                    replicate_folder_path,
                                                    genus_label,
                                                    genus_species_label,
                                                    year_label,
                                                )
                                            )
            
            # Use tqdm to display progress
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing Dataset"):
                self.data.extend(future.result())

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

