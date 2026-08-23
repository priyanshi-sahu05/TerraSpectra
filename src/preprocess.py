import numpy as np
import h5py
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import IncrementalPCA


def get_dataset_info(file_path):
    """Read dataset information without loading the complete data."""
    with h5py.File(file_path, "r") as f:
        data_shape = f["data"].shape
        labels = f["labels"][:]

    print("Original data shape:", data_shape)
    return data_shape, labels


def normalize_batches(file_path, batch_size=10):
    """Fit StandardScaler incrementally using batches."""
    data_shape, _ = get_dataset_info(file_path)
    samples, height, width, bands = data_shape

    scaler = StandardScaler()

    with h5py.File(file_path, "r") as f:
        data = f["data"]

        for start in range(0, samples, batch_size):
            end = min(start + batch_size, samples)

            batch = data[start:end]
            pixels = batch.reshape(-1, bands)

            scaler.partial_fit(pixels)

    print("Incremental normalization completed!")
    return scaler


def fit_pca_batches(file_path, scaler, n_components=20, batch_size=10):
    """Fit IncrementalPCA using batches instead of the whole dataset."""
    data_shape, _ = get_dataset_info(file_path)
    samples, height, width, bands = data_shape

    pca = IncrementalPCA(
        n_components=n_components,
        batch_size=batch_size * height * width
    )

    with h5py.File(file_path, "r") as f:
        data = f["data"]

        for start in range(0, samples, batch_size):
            end = min(start + batch_size, samples)

            batch = data[start:end]
            pixels = batch.reshape(-1, bands)

            normalized = scaler.transform(pixels)
            pca.partial_fit(normalized)

    print("Incremental PCA completed!")
    print("Original spectral bands:", bands)
    print("PCA components:", n_components)

    return pca


def process_and_save_batches(
    input_file,
    output_file,
    scaler,
    pca,
    labels,
    batch_size=10
):
    """Transform and save processed data batch-by-batch."""

    data_shape, _ = get_dataset_info(input_file)
    samples, height, width, bands = data_shape
    n_components = pca.n_components_

    with h5py.File(input_file, "r") as input_h5, \
         h5py.File(output_file, "w") as output_h5:

        input_data = input_h5["data"]

        output_data = output_h5.create_dataset(
            "data",
            shape=(samples, height, width, n_components),
            dtype=np.float32
        )

        output_h5.create_dataset(
            "labels",
            data=labels
        )

        num_batches = (samples + batch_size - 1) // batch_size

        print("Batch processing started")
        print("Total samples:", samples)
        print("Batch size:", batch_size)
        print("Number of batches:", num_batches)

        for i, start in enumerate(range(0, samples, batch_size)):
            end = min(start + batch_size, samples)

            batch = input_data[start:end]
            pixels = batch.reshape(-1, bands)

            normalized = scaler.transform(pixels)
            reduced = pca.transform(normalized)

            processed_batch = reduced.reshape(
                end - start,
                height,
                width,
                n_components
            )

            output_data[start:end] = processed_batch.astype(
                np.float32
            )

            print(
                f"Batch {i + 1}/{num_batches}: "
                f"shape = {processed_batch.shape}"
            )

    print("Processed dataset saved as:", output_file)


def main():
    input_file = "mock_hyperspectral.h5"
    output_file = "processed_hyperspectral.h5"

    batch_size = 10
    n_components = 20

    # 1. Read dataset information
    _, labels = get_dataset_info(input_file)

    # 2. Normalize incrementally
    scaler = normalize_batches(
        input_file,
        batch_size
    )

    # 3. Fit PCA incrementally
    pca = fit_pca_batches(
        input_file,
        scaler,
        n_components,
        batch_size
    )

    # 4. Process and save in batches
    process_and_save_batches(
        input_file,
        output_file,
        scaler,
        pca,
        labels,
        batch_size
    )

    print("Optimized preprocessing completed successfully!")


if __name__ == "__main__":
    main()