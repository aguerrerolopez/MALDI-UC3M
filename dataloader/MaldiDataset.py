from torch.utils.data import Dataset
from dataloader.SpectrumObject import SpectrumObject

class MaldiDataset(Dataset):
    def __init__(self, spectra_dict, preprocess_pipeline=None):
        self.samples = []
        self.preprocess_pipeline = preprocess_pipeline

        for year, genus_dict in spectra_dict.items():
            for genus, species_dict in genus_dict.items():
                for species, studies in species_dict.items():
                    for study_name, fid_paths in studies.items():
                        for fid_path in fid_paths:
                            self.samples.append({
                                'fid': fid_path,
                                'label': f"{genus}_{species}",
                                'meta': {
                                    'year': year,
                                    'genus': genus,
                                    'species': species,
                                    'study': study_name
                                }
                            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        entry = self.samples[idx]
        fid_path = entry['fid']
        acqu_path = fid_path.replace('fid', 'acqu')  # assuming standard Bruker structure

        # Load and preprocess spectrum
        spectrum = SpectrumObject.from_bruker(acqu_path, fid_path)
        if self.preprocess_pipeline:
            spectrum = self.preprocess_pipeline(spectrum)

        # Return SpectrumObject and label
        return SpectrumObject(mz=spectrum.mz, intensity=spectrum.intensity), entry['label'], entry['meta']