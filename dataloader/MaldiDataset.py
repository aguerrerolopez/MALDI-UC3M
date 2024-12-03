import numpy as np
import pandas as pd
import os
import pandas as pd
import numpy as np
from dataloader.preprocess import SequentialPreprocessor, VarStabilizer, Smoother, BaselineCorrecter, Trimmer, Binner, Normalizer
from dataloader.SpectrumObject import SpectrumObject
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


# Code copied from maldi-nn, thanks to the authors. Adapted by Alejandro Guerrero-López.

   
class MaldiDataset:
    """MaldiDataset class. Reads a dataset from a directory containing folders with MALDI-TOF spectra.
    Assumes the following structure:
    root_dir/
        year_folder/
            genus_folder/
                species_folder/
                    replicate_folder/
                        lecture_folder/
                            acqu
                            fid

    Parameters
    ----------
    root_dir : str
        Path to the root directory of the dataset
    preprocess_pipeline : SequentialPreprocessor, optional
        Preprocessing pipeline to apply to the spectra, by default None
    
    Attributes
    ----------
    root_dir : str
        Path to the root directory of the dataset
    preprocess_pipeline : SequentialPreprocessor
        Preprocessing pipeline to apply to the spectra
    data : list
        List of dictionaries containing the spectrum object and labels

    Methods
    -------
    parse_dataset()
        Parse the dataset and store the spectra in the data attribute
    get_data()
        Return the data attribute
    
    """

    def __init__(self, root_dir, preprocess_pipeline:SequentialPreprocessor=None):
        self.root_dir = root_dir
        self.preprocess_pipeline = preprocess_pipeline
        self.data = []

    def parse_dataset(self):
        print(f"Reading dataset from {self.root_dir}")
        # Get total number of folders for progress bar
        total_folders = sum(
            1 for year_folder in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, year_folder))
            for genus_folder in os.listdir(os.path.join(self.root_dir, year_folder)) if os.path.isdir(os.path.join(self.root_dir, year_folder, genus_folder))
            for species_folder in os.listdir(os.path.join(self.root_dir, year_folder, genus_folder)) if os.path.isdir(os.path.join(self.root_dir, year_folder, genus_folder, species_folder))
            for replicate_folder in os.listdir(os.path.join(self.root_dir, year_folder, genus_folder, species_folder)) if os.path.isdir(os.path.join(self.root_dir, year_folder, genus_folder, species_folder, replicate_folder))
        )

        with tqdm(total=total_folders, desc="Processing Dataset", unit="folder") as pbar:
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
                                            for lecture_folder in os.listdir(replicate_folder_path):
                                                lecture_folder_path = os.path.join(replicate_folder_path, lecture_folder)
                                                if os.path.isdir(lecture_folder_path):
                                                    # Search for "acqu" and "fid" files
                                                    acqu_file, fid_file = self._find_acqu_fid_files(lecture_folder_path)
                                                    if acqu_file and fid_file:
                                                        # Read the maldi-tof spectra using from_bruker
                                                        spectrum = SpectrumObject.from_bruker(acqu_file, fid_file)
                                                        # Preprocessing pipeline if any
                                                        if self.preprocess_pipeline:
                                                            spectrum = self.preprocess_pipeline(spectrum)
                                                        # Skip if the spectrum is NaN due to preprocessing
                                                        if np.isnan(spectrum.intensity).any():
                                                            print("Skipping NaN spectrum")
                                                            continue
                                                        self.data.append({
                                                            'spectrum_intensity': spectrum.intensity,
                                                            'spectrum_mz': spectrum.mz,
                                                            'year_label': year_label,
                                                            'genus_label': genus_label,
                                                            'genus_species_label': genus_species_label,
                                                        })
                                            pbar.update(1)


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

    def save_to_hdf5(self, file_name):
        with h5py.File(file_name, 'w') as h5f:
            spectra = np.array([d['spectrum_intensity'] for d in self.data])
            mz_values = np.array([d['spectrum_mz'] for d in self.data])
            year_labels = np.array([d['year_label'] for d in self.data])
            genus_labels = np.array([d['genus_label'] for d in self.data])
            genus_species_labels = np.array([d['genus_species_label'] for d in self.data])

            h5f.create_dataset('spectra', data=spectra, chunks=True, compression='gzip')
            h5f.create_dataset('mz_values', data=mz_values, chunks=True, compression='gzip')
            h5f.create_dataset('year_labels', data=year_labels.astype('S'), chunks=True, compression='gzip')
            h5f.create_dataset('genus_labels', data=genus_labels.astype('S'), chunks=True, compression='gzip')
            h5f.create_dataset('genus_species_labels', data=genus_species_labels.astype('S'), chunks=True, compression='gzip')

    def load_from_hdf5(self, file_name):
        with h5py.File(file_name, 'r') as h5f:
            self.data = [{
                'spectrum_intensity': h5f['spectra'][i],
                'spectrum_mz': h5f['mz_values'][i],
                'year_label': h5f['year_labels'][i].decode('utf-8'),
                'genus_label': h5f['genus_labels'][i].decode('utf-8'),
                'genus_species_label': h5f['genus_species_labels'][i].decode('utf-8'),
            } for i in range(len(h5f['spectra']))]

    def get_data(self):
        return self.data