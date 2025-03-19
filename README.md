# MALDI-UC3M
Base code for all MALDI projects

Structure: 

/MALDI-UC3M/
│── /dataloader/               # Dataset-specific managers live here
│   ├── __init__.py
│   ├── DataAugmenter.py       # Implementation of data augmentation
│   ├── DRIAMS_Manager.py      # Handles DRIAMS dataset
│   ├── MaldiDataset.py        # Reads a dataset from a directory containing folders with MALDI-TOF spectra in BrukerRaw format.
│   ├── SpectrumObject.py      # Handles spectrum parsing
│── /utils/                    # General utilities for all datasets
│   ├── __init__.py
│   ├── preprocess.py          # Preprocessing pipeline
│   ├── spectrum_utils.py      # 
│   ├── visualization.py       # Plots (PCA, t-SNE, spectrum comparisons)
│── /notebooks/                
│   ├── Getting_Started.ipynb  # Plug&Play notebook for starters in MALDI-TOF data analysis
│── LICENSE
│── README.md
│── main.py                    # (Keep) Entry point for batch processing
│── .gitignore