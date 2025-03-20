import random
from dataloader.DRIAMS_Manager import DRIAMS_Dataset 

def split_dataset(dataset, split_by="random", perc=0.2, ood_value=None):
    """
    Splits a DRIAMS_Dataset based on the specified method.
    
    Parameters:
    - dataset (DRIAMS_Dataset): Dataset containing spectra.
    - split_by (str, optional): Criterion for splitting. 
        Options: "random", "hospital", "year".
    - perc (float, optional): Percentage of data to be used for testing (only for "random" split). Default is 0.2.
    - ood_value (str, optional): If splitting by "hospital" or "year", specify the OOD group (e.g., "DRIAMS_D" or "2018").
    
    Returns:
    - train_dataset (DRIAMS_Dataset): Training dataset.
    - test_dataset (DRIAMS_Dataset): Testing dataset (OOD or random split).
    """

    train_spectra = []
    test_spectra = []

    if split_by == "random":
        # Shuffle dataset and split by percentage
        indices = list(range(len(dataset)))
        random.shuffle(indices)
        split_idx = int(len(dataset) * (1 - perc))

        train_spectra = [dataset[i] for i in indices[:split_idx]]
        test_spectra = [dataset[i] for i in indices[split_idx:]]

    elif split_by in ["hospital", "year"]:
        if not ood_value:
            raise ValueError(f"When using split_by='{split_by}', you must provide an ood_value (e.g., 'DRIAMS_D' or '2018').")

        # Split based on hospital or year
        for i in range(len(dataset)):
            spectrum, label = dataset[i]
            metadata = label.split("-")  # Extract metadata (Genus, Species, Hospital, Year)
            
            if split_by == "hospital":
                group = metadata[2]  # Extract hospital
            elif split_by == "year":
                group = metadata[3]  # Extract year

            if group == ood_value:
                test_spectra.append((spectrum, label))
            else:
                train_spectra.append((spectrum, label))

    else:
        raise ValueError("Invalid split_by value. Choose from 'random', 'hospital', or 'year'.")

    # Create new DRIAMS_Dataset objects for training and testing
    train_dataset = DRIAMS_Dataset(dataset.manager, spectra_list=train_spectra)
    test_dataset = DRIAMS_Dataset(dataset.manager, spectra_list=test_spectra)

    return train_dataset, test_dataset