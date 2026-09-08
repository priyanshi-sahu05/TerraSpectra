import os
import h5py
import numpy as np


DATASETS = [
    "processed_hyperspectral.h5",
    "spectral_patches.h5",
]


def test_dataset(file_path):
    print("\nTesting:", file_path)

    if not os.path.exists(file_path):
        print("File not found - skipped")
        return

    with h5py.File(file_path, "r") as f:
        print("Datasets:", list(f.keys()))

        for name in f.keys():
            data = f[name]

            print(f"{name}:")
            print("  Shape:", data.shape)
            print("  Dtype:", data.dtype)

            sample = data[0]
            print("  Sample shape:", sample.shape)

            if np.issubdtype(data.dtype, np.number):
                print("  Finite values:", np.isfinite(sample).all())


def main():
    print("Testing different input datasets...")

    for dataset in DATASETS:
        test_dataset(dataset)

    print("\nInput dataset testing completed successfully!")


if __name__ == "__main__":
    main()