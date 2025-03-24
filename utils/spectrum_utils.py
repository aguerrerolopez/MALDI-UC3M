import random
from dataloader.DRIAMS_Manager import DRIAMS_Dataset 

def split_dataset(dataset, split_by="random", perc=0.2, key=None):
    """
    Splits a DRIAMS_Dataset based on the specified method.
    
    Parameters:
    - dataset (DRIAMS_Dataset): Dataset containing spectra.
    - split_by (str, optional): Criterion for splitting. 
        Options: "random", "hospital", "year".
    - perc (float, optional): Percentage of data to be used for testing (only for "random" split). Default is 0.2.
    - key (str, optional): If splitting by "hospital" or "year", specify the OOD group (e.g., "DRIAMS_D" or "2018").
    
    Returns:
    - train_dataset (DRIAMS_Dataset): Training dataset.
    - test_dataset (DRIAMS_Dataset): Testing dataset (OOD or random split).
    """

    file_list = dataset.file_list
    train_list = []
    test_list = []

    if split_by == "random":
        random.shuffle(file_list)
        split_idx = int(len(file_list) * (1 - perc))
        train_list = file_list[:split_idx]
        test_list = file_list[split_idx:]

    elif split_by in ["hospital", "year"]:
        if not key:
            raise ValueError(f"split_by '{split_by}' requires an key.")

        for file_path, metadata in file_list:
            if split_by == "hospital" and metadata["hospital"] == key:
                test_list.append((file_path, metadata))
            elif split_by == "year" and metadata["year"] == key:
                test_list.append((file_path, metadata))
            else:
                train_list.append((file_path, metadata))
    else:
        raise ValueError("split_by must be one of: 'random', 'hospital', 'year'")

    # Create new dataset instances
    train_dataset = DRIAMS_Dataset(dataset.manager, spectra_list=train_list)
    test_dataset = DRIAMS_Dataset(dataset.manager, spectra_list=test_list)

    return train_dataset, test_dataset