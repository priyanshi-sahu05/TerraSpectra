import h5py
import numpy as np

file_path = "processed_hyperspectral.h5"

with h5py.File(file_path, "r") as f:
    data = f["data"][:]
    labels = f["labels"][:]

print("===== Dataset Verification =====")

print("Data shape:", data.shape)
print("Number of samples:", data.shape[0])
print("Image height:", data.shape[1])
print("Image width:", data.shape[2])
print("Spectral bands/components:", data.shape[3])

print("Missing values:", np.isnan(data).sum())
print("Minimum value:", data.min())
print("Maximum value:", data.max())

print("Labels shape:", labels.shape)
print("Unique labels:", np.unique(labels))

print("Dataset verification completed successfully!")