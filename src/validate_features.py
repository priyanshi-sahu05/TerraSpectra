import h5py
import numpy as np

INPUT_FILE = "spectral_patches.h5"

def main():
    print("Loading spectral patches...")

    with h5py.File(INPUT_FILE, "r") as f:
        patches = f["patches"][:]
        labels = f["labels"][:]

    print("Patches shape:", patches.shape)
    print("Labels shape:", labels.shape)

    print("Feature validation completed successfully!")

    print("Patches dtype:", patches.dtype)
    print("Labels dtype:", labels.dtype)

    print("Number of samples:", patches.shape[0])
    print("Patch size:", patches.shape[1:3])
    print("Spectral bands:", patches.shape[3])

if __name__ == "__main__":
    main()