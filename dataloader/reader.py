import numpy as np
import pandas as pd
from scipy.stats import binned_statistic
import os
from scipy.signal import savgol_filter
from scipy import sparse
from scipy.linalg import norm
import pandas as pd
import numpy as np
from scipy.stats import binned_statistic
import matplotlib.pyplot as plt
from preprocess import SequentialPreprocessor, VarStabilizer, Smoother, BaselineCorrecter, Trimmer, Binner, Normalizer



# Code copied from maldi-nn, thanks to the authors. Adapted by Alejandro Guerrero-López.


class SpectrumObject:
    """Base Spectrum Object class

    Can be instantiated directly with 1-D np.arrays for mz and intensity.
    Alternatively, can be read from csv files or from bruker output data.
    Reading from Bruker data is based on the code in https://github.com/sgibb/readBrukerFlexData

    Parameters
    ----------
    mz : 1-D np.array, optional
        mz values, by default None
    intensity : 1-D np.array, optional
        intensity values, by default None
    """

    def __init__(self, mz=None, intensity=None):
        self.mz = mz
        self.intensity = intensity
        if self.intensity is not None:
            if np.issubdtype(self.intensity.dtype, np.unsignedinteger):
                self.intensity = self.intensity.astype(int)
        if self.mz is not None:
            if np.issubdtype(self.mz.dtype, np.unsignedinteger):
                self.mz = self.mz.astype(int)

    def __getitem__(self, index):
        return SpectrumObject(mz=self.mz[index], intensity=self.intensity[index])

    def __len__(self):
        if self.mz is not None:
            return self.mz.shape[0]
        else:
            return 0

    def plot(self, as_peaks=False, **kwargs):
        """Plot a spectrum via matplotlib

        Parameters
        ----------
        as_peaks : bool, optional
            draw points in the spectrum as individualpeaks, instead of connecting the points in the spectrum, by default False
        """
        if as_peaks:
            mz_plot = np.stack([self.mz - 1, self.mz, self.mz + 1]).T.reshape(-1)
            int_plot = np.stack(
                [
                    np.zeros_like(self.intensity),
                    self.intensity,
                    np.zeros_like(self.intensity),
                ]
            ).T.reshape(-1)
        else:
            mz_plot, int_plot = self.mz, self.intensity
        plt.plot(mz_plot, int_plot, **kwargs)

    def __repr__(self):
        string_ = np.array2string(
            np.stack([self.mz, self.intensity]), precision=5, threshold=10, edgeitems=2
        )
        mz_string, int_string = string_.split("\n")
        mz_string = mz_string[1:]
        int_string = int_string[1:-1]
        return "SpectrumObject([\n\tmz  = %s,\n\tint = %s\n])" % (mz_string, int_string)

    @staticmethod
    def tof2mass(ML1, ML2, ML3, TOF):
        A = ML3
        B = np.sqrt(1e12 / ML1)
        C = ML2 - TOF

        if A == 0:
            return (C * C) / (B * B)
        else:
            return ((-B + np.sqrt((B * B) - (4 * A * C))) / (2 * A)) ** 2

    @classmethod
    def from_bruker(cls, acqu_file, fid_file):
        """Read a spectrum from Bruker's format

        Parameters
        ----------
        acqu_file : str
            "acqu" file bruker folder
        fid_file : str
            "fid" file in bruker folder

        Returns
        -------
        SpectrumObject
        """
        with open(acqu_file, "rb") as f:
            lines = [line.decode("utf-8", errors="replace").rstrip() for line in f]
        for l in lines:
            if l.startswith("##$TD"):
                TD = int(l.split("= ")[1])
            if l.startswith("##$DELAY"):
                DELAY = int(l.split("= ")[1])
            if l.startswith("##$DW"):
                DW = float(l.split("= ")[1])
            if l.startswith("##$ML1"):
                ML1 = float(l.split("= ")[1])
            if l.startswith("##$ML2"):
                ML2 = float(l.split("= ")[1])
            if l.startswith("##$ML3"):
                ML3 = float(l.split("= ")[1])
            if l.startswith("##$BYTORDA"):
                BYTORDA = int(l.split("= ")[1])
            if l.startswith("##$NTBCal"):
                NTBCal = l.split("= ")[1]

        intensity = np.fromfile(fid_file, dtype={0: "<i", 1: ">i"}[BYTORDA])

        if len(intensity) < TD:
            TD = len(intensity)
        TOF = DELAY + np.arange(TD) * DW

        mass = cls.tof2mass(ML1, ML2, ML3, TOF)

        intensity[intensity < 0] = 0

        return cls(mz=mass, intensity=intensity)

    @classmethod
    def from_tsv(cls, file, sep=" "):
        """Read a spectrum from txt

        Parameters
        ----------
        file : str
            path to csv file
        sep : str, optional
            separator in the file, by default " "

        Returns
        -------
        SpectrumObject
        """
        s = pd.read_table(
            file, sep=sep, index_col=None, comment="#", header=None
        ).values
        mz = s[:, 0]
        intensity = s[:, 1]
        return cls(mz=mz, intensity=intensity)

   
class MaldiDataset:
    def __init__(self, root_dir, n_step=3):
        self.root_dir = root_dir
        self.n_step = n_step
        self.data = []

    def parse_dataset(self):
        for folder in os.listdir(self.root_dir):
            folder_path = os.path.join(self.root_dir, folder)
            if os.path.isdir(folder_path):
                # Parse folder name for genus, species, and hospital code
                genus_species, hospital_code = self._parse_folder_name(folder)
                genus = genus_species.split()[0]
                genus_species_label = genus_species
                unique_id_label = hospital_code

                # Iterate over each replicate folder
                for replicate_folder in os.listdir(folder_path):
                    replicate_folder_path = os.path.join(folder_path, replicate_folder)
                    if os.path.isdir(replicate_folder_path):
                        # Iterate over each lecture folder
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
                                    # Append data point to the dataset
                                    # if the spectrum is nan due to the preprocessing, skip it
                                    if np.isnan(binned_spectrum.intensity).any():
                                        print("Skipping nan spectrum")
                                        continue
                                    self.data.append({
                                        'spectrum': binned_spectrum.intensity,
                                        'm/z': binned_spectrum.mz,
                                        'unique_id_label': unique_id_label,
                                        'genus_label': genus,
                                        'genus_species_label': genus_species_label
                                    })

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

