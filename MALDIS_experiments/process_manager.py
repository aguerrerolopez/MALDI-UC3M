import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dataloader.MaldiMaranon_Manager import MaldiMaranonManager


# MALDIMARANON dataset
dataset_path = f"/export/data_ml4ds/bacteria_id/MaldiMaranonDB"

# Initialize the DRIAMS manager
manager = MaldiMaranonManager(dataset_path)
pickle_path = os.path.join(os.path.dirname(__file__), 'maldi_manager.pkl')
manager.save_to_pickle(pickle_path)

# Get statistics and save to CSV
stats_df = manager.stats
stats_df.to_csv("maldi_statistics.csv")