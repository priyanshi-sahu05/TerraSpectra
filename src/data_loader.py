import h5py
import numpy as np


def load_dataset(file_path):
    """Load processed hyperspectral dataset from HDF5 file."""

    with h5py.File(file_path, "r") as f:
        data = f["data"][:]
        labels = f["labels"][:]

    return data, labels


if __name__ == "__main__":

    file_path = "processed_hyperspectral.h5"

    data, labels = load_dataset(file_path)

    print("===== Data Loader =====")
    print("Data loaded successfully!")
    print("Data shape:", data.shape)
    print("Labels shape:", labels.shape)
    print("Number of samples:", len(labels))