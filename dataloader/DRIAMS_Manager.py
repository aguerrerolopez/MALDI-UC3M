import os
import numpy as np
import shutil
import pandas as pd
import logging
from datetime import datetime
from collections import defaultdict
from dataloader.SpectrumObject import SpectrumObject

def clean_database(original_path, destination_path):
    """
    Cleans and restructures the DRIAMS dataset by:
    - Extracting raw spectra files (`.txt`) from the dataset.
    - Organizing them into a new hierarchical structure based on:
        - Dataset (DRIAMS-A, DRIAMS-B, etc.)
        - Year (2015, 2016, 2017, 2018)
        - Genus and Species
    - Copies the extracted raw spectra into the structured output directory.
    - Logs the entire process, including missing files and successful copies.

    Parameters:
        original_path (str): The path to the original DRIAMS dataset.
        destination_path (str): The path where the cleaned and restructured dataset will be saved.
    Returns:
        None
    """
    # Set up logging
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join(destination_path, "logs")
    os.makedirs(log_dir, exist_ok=True)  # Ensure logs directory exists
    log_filename = os.path.join(log_dir, f'process_{timestamp}.log')

    logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s - %(message)s')
    logging.info("Starting DRIAMS raw spectra processing...")

    # Define the datasets to process
    datasets = [d for d in os.listdir(original_path) if d.startswith("DRIAMS-")]
    
    for dataset in datasets:
        raw_dir = os.path.join(original_path, dataset, "raw")  # Path to raw spectra
        csv_dir = os.path.join(original_path, dataset, "id")  # Path to metadata CSVs
        output_dir = os.path.join(destination_path, dataset)  # Destination structure

        # Iterate over all available years
        for year in ["2015", "2016", "2017", "2018"]:
            csv_file = os.path.join(csv_dir, year, f'{year}_clean.csv')  # Metadata file
            year_path = os.path.join(raw_dir, year)  # Year-specific raw data folder

            # Check if the raw folder for that year exists
            if os.path.exists(year_path):
                if not os.path.exists(csv_file):
                    logging.warning(f"Missing metadata CSV for {dataset} {year}. Skipping...")
                    continue

                # Read metadata CSV to associate IDs with species names
                df = pd.read_csv(csv_file)
                for _, row in df.iterrows():
                    identifier = row['code']
                    print(f'Processing {identifier}...')

                    # Extract genus and species from the species column
                    genus_species = row['species'].split(' ')
                    if len(genus_species) < 2:
                        logging.warning(f"Skipping {identifier}: Invalid species format.")
                        continue

                    genus = genus_species[0].capitalize()
                    if genus.startswith('Mix!'):  # Handle "Mix!" prefix cases
                        genus = genus[4:].capitalize()
                    species = genus_species[1].split('[')[0].capitalize()

                    # Define file paths
                    txt_file = os.path.join(raw_dir, year, f'{identifier}.txt')
                    print(f'Copying {txt_file} to {output_dir}/{year}/{genus}/{species}...')

                    # Check if the spectrum file exists before copying
                    if os.path.exists(txt_file):
                        output_path = os.path.join(output_dir, year, genus, species)
                        output_path_TOTAL = os.path.join(output_dir, 'TOTAL', genus, species)

                        # Ensure directories exist before copying files
                        os.makedirs(output_path, exist_ok=True)
                        os.makedirs(output_path_TOTAL, exist_ok=True)

                        # Copy the spectrum files into both specific and TOTAL directories
                        shutil.copy(txt_file, output_path)
                        shutil.copy(txt_file, output_path_TOTAL)

                        # Log the successful copy
                        logging.info(f'✅ Copied {txt_file} to {output_path}')
                        logging.info(f'✅ Copied {txt_file} to {output_path_TOTAL}')
                    else:
                        logging.warning(f'❌ Missing spectrum file: {txt_file}')

    logging.info("Raw spectra processing completed successfully!")

class DRIAMS_Manager:
    """
    A class to load, manage, and query the entire DRIAMS dataset.

    Attributes:
    - dataset_path (str): The root path to the processed DRIAMS dataset.
    - files_dict (dict): A nested dictionary mapping (dataset, year, genus, species) to file paths.

    Methods:
    - load_dataset(): Scans the dataset and stores file paths.
    - get_statistics(): Generates a Pandas DataFrame summarizing species occurrences across all datasets.
    - query_spectra(): Retrieves spectrum files based on dataset, year, genus, and/or species conditions.
    - get_taxonomy(): Retrieves bacterial taxonomy information based on given filters.
    """

    def __init__(self, dataset_path):
        """
        Initializes the dataset object and loads all available spectrum files.

        Parameters:
        - dataset_path (str): The root directory of the processed DRIAMS dataset.
        """
        self.dataset_path = dataset_path
        self.spectra_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
        self.load_spectra_paths()  # Populate self.spectra_dict
        self.stats = self.compute_statistics()  # Precompute statistics

    def load_spectra_paths(self):
        """
        Loads all spectra paths into a structured dictionary:
        self.spectra_dict[hospital][year][genus][species] -> List of file paths.
        """
        if not os.path.exists(self.dataset_path):
            raise ValueError(f"⚠️ The dataset path does not exist: {self.dataset_path}")

        # Scan all DRIAMS folders (e.g., DRIAMS_A, DRIAMS_B, ...)
        for dataset in os.listdir(self.dataset_path):
            dataset_path = os.path.join(self.dataset_path, dataset)
            if not os.path.isdir(dataset_path) or not dataset.startswith("DRIAMS_"):
                continue

            # Scan all years inside each dataset
            for year in os.listdir(dataset_path):
                year_path = os.path.join(dataset_path, year)
                if not os.path.isdir(year_path) or not year.isdigit():
                    continue  # Skip non-year folders

                # Scan all genus, species, and nested species folders
                for genus in os.listdir(year_path):
                    genus_path = os.path.join(year_path, genus)
                    if not os.path.isdir(genus_path):
                        continue

                    for species in os.listdir(genus_path):
                        species_path = os.path.join(genus_path, species)
                        if not os.path.isdir(species_path):
                            continue

                        # Store all spectrum files for the given dataset/year/genus/species
                        for file in os.listdir(species_path):
                            if file.endswith(".txt"):
                                file_path = os.path.join(species_path, file)
                                self.spectra_dict[dataset][year][genus][species].append(file_path)

    def compute_statistics(self):
        """
        Computes dataset statistics: Number of spectra per hospital/year/genus/species.

        Returns:
        - DataFrame summarizing spectra counts.
        """
        species_counts = defaultdict(lambda: defaultdict(int))

        for dataset in self.spectra_dict:
            for year in self.spectra_dict[dataset]:
                for genus in self.spectra_dict[dataset][year]:
                    for species in self.spectra_dict[dataset][year][genus]:
                        count = len(self.spectra_dict[dataset][year][genus][species])
                        species_counts[(genus, species)][f"{dataset}_{year}"] = count
                        species_counts[(genus, species)]["TOTAL"] += count

        df_stats = pd.DataFrame.from_dict(species_counts, orient="index").fillna(0).astype(int)
        df_stats.index = pd.MultiIndex.from_tuples(df_stats.index, names=["GENUS", "SPECIES"])
        df_stats = df_stats.reset_index()

        # Move TOTAL column to last
        column_order = ["GENUS", "SPECIES"] + sorted([col for col in df_stats.columns if col not in ["GENUS", "SPECIES", "TOTAL"]]) + ["TOTAL"]
        df_stats = df_stats[column_order]

        return df_stats
    
    def get_taxonomy(self, dataset=None, year=None, genus=None, return_species=False):
        """
        Retrieves bacterial taxonomy information based on given filters.

        Parameters:
        - dataset (str, optional): The hospital dataset (e.g., "DRIAMS_A").
        - year (str, optional): The year to filter (e.g., "2018").
        - genus (str, optional): The bacterial genus to filter (e.g., "Shigella").
        - return_species (bool, optional): 
            - If False, returns only genus names.
            - If True and genus is provided, returns a list of species under that genus.
            - If True and genus is NOT provided, returns (genus, species) pairs.

        Returns:
        - List of unique genus names, species names, or (genus, species) pairs.
        """
        taxonomy_set = set()

        # Iterate through datasets
        for d in self.spectra_dict:
            if dataset and d != dataset:
                continue

            for y in self.spectra_dict[d]:
                if year and y != year:
                    continue

                for g in self.spectra_dict[d][y]:
                    if genus:
                        # If filtering by genus, return species or empty if genus not found
                        if g.lower() == genus.lower():
                            if return_species:
                                taxonomy_set.update(self.spectra_dict[d][y][g].keys())  # Return only species names
                            else:
                                taxonomy_set.add(g)  # Return genus name
                    else:
                        # If NOT filtering by genus, return genus OR (genus, species) pairs
                        if return_species:
                            for s in self.spectra_dict[d][y][g]:
                                taxonomy_set.add((g, s))  # Store (genus, species)
                        else:
                            taxonomy_set.add(g)  # Store genus name

        return sorted(taxonomy_set)  # Return sorted list
                            
    def query_spectra(self, genus=None, species=None, hospital=None, year=None):
        """
        Retrieves file paths matching the given filters.

        Parameters:
        - genus (str or list, optional): Filter by genus.
        - species (str or list, optional): Filter by species.
        - hospital (str or list, optional): Filter by hospital.
        - year (str or list, optional): Filter by year.

        Returns:
        - List of tuples: (file_path, metadata_dict), where metadata_dict contains:
        {'genus': genus, 'species': species, 'hospital': hospital, 'year': year}
        """
        spectra_list = []

        hospitals = hospital if hospital else self.spectra_dict.keys()
        hospitals = [hospitals] if isinstance(hospitals, str) else hospitals # check if hospital is a list, if it is a string, create a list

        for hosp in hospitals:
            if hosp not in self.spectra_dict:
                continue

            years = year if year else self.spectra_dict[hosp].keys()
            years = [years] if isinstance(years, str) else years
            for y in years:
                if y not in self.spectra_dict[hosp]:
                    continue

                genera = genus if genus else self.spectra_dict[hosp][y].keys()
                genera = [genera] if isinstance(genera, str) else genera
                for g in genera:
                    if g not in self.spectra_dict[hosp][y]:
                        continue

                    species_list = species if species else self.spectra_dict[hosp][y][g].keys()
                    species_list = [species_list] if isinstance(species_list, str) else species_list
                    for s in species_list:
                        if s in self.spectra_dict[hosp][y][g]:
                            for file_path in self.spectra_dict[hosp][y][g][s]:
                                # Store file path + metadata
                                metadata = {"genus": g, "species": s, "hospital": hosp, "year": y}
                                spectra_list.append((file_path, metadata))

        return spectra_list
    
    def load_spectrum(self, file_path):
        """
        Loads a single spectrum from a TXT file.

        Parameters:
        - file_path (str): Path to the spectrum file.

        Returns:
        - SpectrumObject instance.
        """
        return SpectrumObject.from_tsv(file_path)
    
class DRIAMS_Dataset:
    """
    Efficiently loads spectra from the DRIAMS dataset for ML training.
    """

    def __init__(self, manager, genus, species, hospital=None, year=None):
        """
        Initializes dataset by retrieving spectra paths.

        Parameters:
        - manager (DRIAMS_Manager): Instance of DRIAMS_Manager.
        - genus (str): Bacterial genus.
        - species (str): Bacterial species.
        - hospital (str, optional): Filter by hospital.
        - year (str, optional): Filter by year.
        """
        self.manager = manager
        self.file_list = self.manager.query_spectra(genus, species, hospital, year)
        self.labels = self.generate_labels()

    def __len__(self):
        """Returns the number of spectra in the dataset."""
        return len(self.file_list)

    def __getitem__(self, index):
        """
        Lazily loads a single spectrum.

        Returns:
        - Tuple (m/z values, intensity values, label)
        """
        file_path, metadata = self.file_list[index]
        spectrum = self.manager.load_spectrum(file_path)

        # Construct label: Genus-Species-Hospital-Year
        label = f"{metadata['genus']}-{metadata['species']}-{metadata['hospital']}-{metadata['year']}"

        return SpectrumObject(mz=spectrum.mz, intensity=spectrum.intensity), label
    
    def generate_labels(self):
        """
        Generates labels in the format `Genus-Species-Hospital-Year` for each sample.

        Returns:
        - List of labels.
        """
        labels = []
        for _, metadata in self.file_list:
            labels.append(f"{metadata['genus']}-{metadata['species']}-{metadata['hospital']}-{metadata['year']}")
        return labels

    def get_feature_matrix(self):
        """
        Loads all spectra and returns a feature matrix.

        Returns:
        - X (numpy array): Feature matrix (intensity values).
        - y (numpy array): Labels.
        """
        # Extract only file paths from query results (ignore metadata)
        file_paths = [entry[0] if isinstance(entry, tuple) else entry for entry in self.file_list]

        # Load spectra
        X = [self.manager.load_spectrum(f).intensity for f in file_paths]
        
        return np.array(X), np.array(self.labels)