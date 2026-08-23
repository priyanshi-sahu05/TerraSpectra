import numpy as np
import h5py
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def load_dataset(file_path):
    """Load hyperspectral data and labels from HDF5 file."""
    with h5py.File(file_path, "r") as f:
        data = f["data"][:]
        labels = f["labels"][:]

    print("Dataset loaded successfully!")
    print("Original data shape:", data.shape)

    return data, labels


def normalize_data(data):
    """Normalize hyperspectral spectral features."""
    samples, height, width, bands = data.shape

    pixels = data.reshape(-1, bands)

    scaler = StandardScaler()
    normalized = scaler.fit_transform(pixels)

    normalized = normalized.reshape(samples, height, width, bands)

    print("Normalization completed successfully!")
    print("Normalized shape:", normalized.shape)

    return normalized


def apply_pca(data, n_components=20):
    """Reduce spectral bands using PCA."""
    samples, height, width, bands = data.shape

    pixels = data.reshape(-1, bands)

    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(pixels)

    reduced = reduced.reshape(samples, height, width, n_components)

    print("PCA completed successfully!")
    print("After PCA:", reduced.shape)

    return reduced


def save_processed_data(file_path, data, labels):
    """Save processed hyperspectral data."""
    with h5py.File(file_path, "w") as f:
        f.create_dataset("data", data=data.astype(np.float32))
        f.create_dataset("labels", data=labels)

    print("Processed dataset saved as:", file_path)

def process_in_batches(data, batch_size=10):
    """Process dataset in smaller batches to reduce memory usage."""

    total_samples = data.shape[0]
    num_batches = (total_samples + batch_size - 1) // batch_size

    print("Batch processing started")
    print("Total samples:", total_samples)
    print("Batch size:", batch_size)
    print("Number of batches:", num_batches)

    for i in range(num_batches):
        start = i * batch_size
        end = min(start + batch_size, total_samples)

        batch = data[start:end]

        print(
            f"Batch {i + 1}/{num_batches}: "
            f"shape = {batch.shape}"
        )

    print("Batch processing completed successfully!")

def main():
    input_file = "mock_hyperspectral.h5"
    output_file = "processed_hyperspectral.h5"

    # 1. Load dataset
    data, labels = load_dataset(input_file)

    # 2. Normalize
    normalized = normalize_data(data)

    # 3. PCA
    processed = apply_pca(normalized, n_components=20)

    process_in_batches(processed, batch_size=10)

    # 4. Save processed dataset
    save_processed_data(output_file, processed, labels)

    print("Preprocessing pipeline completed successfully!")


if __name__ == "__main__":
    main()