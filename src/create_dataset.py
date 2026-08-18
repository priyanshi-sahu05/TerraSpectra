import numpy as np
import h5py

# Mock hyperspectral dataset
samples = 100
height = 32
width = 32
bands = 200

# Random hyperspectral data
data = np.random.rand(samples, height, width, bands).astype(np.float32)

# Random labels: 0 = healthy, 1 = diseased
labels = np.random.randint(0, 2, size=samples)

# Save dataset
with h5py.File("mock_hyperspectral.h5", "w") as f:
    f.create_dataset("data", data=data)
    f.create_dataset("labels", data=labels)

print("Mock hyperspectral dataset created successfully!")
print("Data shape:", data.shape)
print("Labels shape:", labels.shape)