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

    print("\nChecking preprocessing edge cases...")

    # Empty dataset check
    if patches.size == 0:
        print("ERROR: Patch dataset is empty!")
    else:
        print("PASS: Patch dataset is not empty.")

    # Shape consistency
    if patches.shape[0] == labels.shape[0]:
        print("PASS: Patch and label counts match.")
    else:
        print("ERROR: Patch and label counts do not match!")

    # NaN check
    if np.isnan(patches).any():
        print("ERROR: NaN values found!")
    else:
        print("PASS: No NaN values found.")

    # Infinite value check
    if np.isinf(patches).any():
        print("ERROR: Infinite values found!")
    else:
        print("PASS: No infinite values found.")

    # Dimension check
    if patches.ndim == 4:
        print("PASS: Patch data has expected 4 dimensions.")
    else:
        print("ERROR: Unexpected patch dimensions!")

    # Label check
    if labels.ndim == 1:
        print("PASS: Labels have expected 1D shape.")
    else:
        print("ERROR: Unexpected label shape!")

    print("\nPreprocessing edge-case validation completed!")


if __name__ == "__main__":
    main()