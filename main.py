from dataloader.reader import MaldiDataset


data_path = "/export/data_ml4ds/bacteria_id/RAW_MaldiMaranon/data_cleaner_results_v2/MaldiMaranonDB"

dataset = MaldiDataset(data_path, n_step=3)

dataset.parse_dataset()

