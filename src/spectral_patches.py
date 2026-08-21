import h5py
import numpy as np


INPUT_FILE = "processed_hyperspectral.h5"
OUTPUT_FILE = "spectral_patches.h5"

PATCH_SIZE = 5


def extract_patches(data, labels, patch_size=5):
    """
    Generate model-ready spatial-spectral patches.
    """

    samples, height, width, bands = data.shape

    margin = patch_size // 2

    patches = []
    patch_labels = []

    for sample in range(samples):

        for row in range(margin, height - margin):
            for col in range(margin, width - margin):

                patch = data[
                    sample,
                    row - margin:row + margin + 1,
                    col - margin:col + margin + 1,
                    :
                ]

                patches.append(patch)
                patch_labels.append(labels[sample])

    patches = np.asarray(patches, dtype=np.float32)
    patch_labels = np.asarray(patch_labels)

    return patches, patch_labels


def main():

    print("Loading processed dataset...")

    with h5py.File(INPUT_FILE, "r") as f:
        data = f["data"][:]
        labels = f["labels"][:]

    print("Input data shape:", data.shape)
    print("Labels shape:", labels.shape)

    print("Generating spectral patches...")

    patches, patch_labels = extract_patches(
        data,
        labels,
        PATCH_SIZE
    )

    print("Patch generation completed!")
    print("Patches shape:", patches.shape)
    print("Patch labels shape:", patch_labels.shape)

    print("Saving model-ready patches...")

    with h5py.File(OUTPUT_FILE, "w") as f:
        f.create_dataset("patches", data=patches)
        f.create_dataset("labels", data=patch_labels)

    print("Spectral patches saved as:", OUTPUT_FILE)


if __name__ == "__main__":
    main()