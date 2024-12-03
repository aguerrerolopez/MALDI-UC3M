from dataloader.MaldiDataset import MaldiDataset
from dataloader.preprocess import SequentialPreprocessor, VarStabilizer, Smoother, BaselineCorrecter, Trimmer, Binner, Normalizer
from dataloader.DataAugmenter import DataAugmenter


data_path = "/export/data_ml4ds/bacteria_id/RAW_MaldiMaranon/data_cleaner_results_v2/MaldiMaranonDB"

# Preprocessing pipeline
binning_step = 3
preprocess_pipeline = SequentialPreprocessor(VarStabilizer(method="sqrt"),
                                Smoother(halfwindow=10),
                                BaselineCorrecter(method="SNIP", snip_n_iter=20),
                                Trimmer(),
                                Binner(step=binning_step),
                                Normalizer(sum=1)
                                )

# Data augmentation
data_augmenter = DataAugmenter() #TODO: Implement data augmentation

# Create dataset
dataset = MaldiDataset(data_path, preprocess_pipeline)

dataset.parse_dataset()

# Get spectral data
spectra = dataset.get_data()['spectrum']