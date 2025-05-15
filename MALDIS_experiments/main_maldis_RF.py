import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay, accuracy_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dataloader.MaldiMaranon_Manager import MaldiMaranonManager
from dataloader.MaldiDataset import MaldiDataset
from utils.preprocess import SequentialPreprocessor, VarStabilizer, Smoother, BaselineCorrecter, Trimmer, Binner, Normalizer, StdThresholder, MinMaxScaler

def main():

    # ------------------------------
    # 1) SETUP: data, hyperparams
    # ------------------------------

    data_name = 'MALDIS'
    name = 'rf'
    result_dir = f'results/{data_name}_{name}_{time.strftime("%Y%m%d_%H%M%S")}/'
    os.makedirs(result_dir, exist_ok=True)

   # MALDIMARANON dataset
    # Load full training dataset
    dataset_path = f"/export/data_ml4ds/bacteria_id/MaldiMaranonDB"

    # Initialize the DRIAMS manager
    pickle_path = os.path.join(os.path.dirname(__file__), 'maldi_manager.pkl')
    manager = MaldiMaranonManager(dataset_path, presaved=True, pickel_path=pickle_path)

    top_species = manager.get_top_species(top_n=5)
    

    # ---------------- PREPROCESSING COMPARISON -------------------
    binning_step = 9
    processing_basic = SequentialPreprocessor(VarStabilizer(method="sqrt"),
                                        Smoother(halfwindow=10),
                                        BaselineCorrecter(method="SNIP", snip_n_iter=20),
                                        Trimmer(),
                                        Binner(step=binning_step),
                                        MinMaxScaler())
                                        # Normalizer(sum=1))
    
    processing_stdthr = SequentialPreprocessor(VarStabilizer(method="sqrt"),
                                        Smoother(halfwindow=10),
                                        BaselineCorrecter(method="SNIP", snip_n_iter=20),
                                        StdThresholder(factor=1.0),
                                        Trimmer(),
                                        Binner(step=binning_step),
                                        MinMaxScaler())
                                        # Normalizer(sum=1))
    

    # ---------------- DATASET CREATION -------------------

    # Prepare basic and preprocessed datasets
    train_years = ['2018', '2019', '2020', '2021', '2022']
    test_years = ['2023', '2024']

    X_basic_train, y_basic_train = [], []
    X_stdthr_train, y_stdthr_train = [], []

    X_basic_test, y_basic_test = [], []
    X_stdthr_test, y_stdthr_test = [], []

    for genus, species in top_species:
        spectra_dict_train = manager.query_spectra_dict(years=train_years, genus=genus, species=species)
        spectra_dict_test = manager.query_spectra_dict(years=test_years, genus=genus, species=species)

        basic_dataset_train = MaldiDataset(spectra_dict_train, preprocess_pipeline=processing_basic, visualize=True, path=result_dir)
        stdthr_dataset_train = MaldiDataset(spectra_dict_train, preprocess_pipeline=processing_stdthr, visualize=True, path=result_dir)

        basic_dataset_test = MaldiDataset(spectra_dict_test, preprocess_pipeline=processing_basic, visualize=True, path=result_dir)
        stdthr_dataset_test = MaldiDataset(spectra_dict_test, preprocess_pipeline=processing_stdthr, visualize=True, path=result_dir)

        for spectrum, label, _ in basic_dataset_train:
            X_basic_train.append(spectrum.intensity)
            y_basic_train.append(label)

        for spectrum, label, _ in stdthr_dataset_train:
            X_stdthr_train.append(spectrum.intensity)
            y_stdthr_train.append(label)

        for spectrum, label, _ in basic_dataset_test:
            X_basic_test.append(spectrum.intensity)
            y_basic_test.append(label)
        
        for spectrum, label, _ in stdthr_dataset_test:
            X_stdthr_test.append(spectrum.intensity)
            y_stdthr_test.append(label)

    # Convert to numpy
    X_basic_train = np.stack(X_basic_train)
    X_stdthr_train = np.stack(X_stdthr_train)

    # Shuffle train sets
    X_basic_train, y_basic_train = shuffle(X_basic_train, y_basic_train, random_state=42)
    X_stdthr_train, y_stdthr_train = shuffle(X_stdthr_train, y_stdthr_train, random_state=42)

    # Shuffle test sets
    X_basic_test, y_basic_test = shuffle(X_basic_test, y_basic_test, random_state=42)
    X_stdthr_test, y_stdthr_test = shuffle(X_stdthr_test, y_stdthr_test, random_state=42)


    # ---------------- RF TRAINING -------------------

    # Train classifiers
    print("Training classifiers...")
    clf_basic = RandomForestClassifier(random_state=42).fit(X_basic_train, y_basic_train)
    clf_stdthr = RandomForestClassifier(random_state=42).fit(X_stdthr_train, y_stdthr_train)


    # ---------------- MODEL SAVING -------------------

    joblib.dump(clf_basic, os.path.join(result_dir, "rf_basic_model.pkl"))
    joblib.dump(clf_stdthr, os.path.join(result_dir, "rf_stdthr_model.pkl"))


    # ---------------- EVALUATION -------------------

    acc_basic = accuracy_score(y_basic_test, clf_basic.predict(X_basic_test))
    acc_stdthr = accuracy_score(y_stdthr_test, clf_stdthr.predict(X_stdthr_test))

    print(f"Accuracy with basic preprocessing: {acc_basic:.4f}")
    print(f"Accuracy with stdthr preprocessing   : {acc_stdthr:.4f}")

    # Confusion matrices
    cm_basic = confusion_matrix(y_basic_test, clf_basic.predict(X_basic_test), labels=clf_basic.classes_)
    cm_stdthr = confusion_matrix(y_stdthr_test, clf_stdthr.predict(X_stdthr_test), labels=clf_stdthr.classes_)

    # Plot and save confusion matrices
    _, ax1 = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay(cm_basic, display_labels=clf_basic.classes_).plot(ax=ax1, xticks_rotation=45)
    plt.title("Confusion Matrix - Basic Preprocessing")
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, "confusion_matrix_basic.png"))

    _, ax2 = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay(cm_stdthr, display_labels=clf_stdthr.classes_).plot(ax=ax2, xticks_rotation=45)
    plt.title("Confusion Matrix - With StdThresholder")
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, "confusion_matrix_stdthr.png"))

    # Classification reports
    report_basic = classification_report(y_basic_test, clf_basic.predict(X_basic_test), target_names=clf_basic.classes_)
    report_stdthr = classification_report(y_stdthr_test, clf_stdthr.predict(X_stdthr_test), target_names=clf_stdthr.classes_)

    with open(os.path.join(result_dir, "classification_report_basic.txt"), "w") as f:
        f.write(report_basic)

    with open(os.path.join(result_dir, "classification_report_stdthr.txt"), "w") as f:
        f.write(report_stdthr)

if __name__ == "__main__":
    main()