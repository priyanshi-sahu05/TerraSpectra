import numpy as np
import h5py
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load mock hyperspectral dataset
with h5py.File("mock_hyperspectral.h5", "r") as f:
    data = f["data"][:]
    labels = f["labels"][:]

print("Original data shape:", data.shape)

# Shape: (samples, height, width, bands)
samples, height, width, bands = data.shape

# Reshape for preprocessing
pixels = data.reshape(-1, bands)

# Normalization
scaler = StandardScaler()
normalized = scaler.fit_transform(pixels)

# PCA - reduce 200 spectral bands to 20 components
pca = PCA(n_components=20)
reduced = pca.fit_transform(normalized)

# Reshape back
processed_data = reduced.reshape(samples, height, width, 20)

# Batch processing
batch_size = 10
total_samples = processed_data.shape[0]
num_batches = (total_samples + batch_size - 1) // batch_size

print("Batch processing started")
print("Total samples:", total_samples)
print("Batch size:", batch_size)
print("Number of batches:", num_batches)

for i in range(num_batches):
    start = i * batch_size
    end = min(start + batch_size, total_samples)

    batch = processed_data[start:end]

    print(f"Batch {i + 1}/{num_batches}: shape = {batch.shape}")

print("Batch processing completed successfully!")

print("After PCA:", processed_data.shape)

# Save processed dataset
with h5py.File("processed_hyperspectral.h5", "w") as f:
    f.create_dataset("data", data=processed_data.astype(np.float32))
    f.create_dataset("labels", data=labels)

print("Normalization and PCA completed successfully!")
print("Processed dataset saved as: processed_hyperspectral.h5")