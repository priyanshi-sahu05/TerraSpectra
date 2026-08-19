# TerraSpectra – Preprocessing Documentation

## 1. Dataset
A mock hyperspectral dataset is used for the TerraSpectra data pipeline.

- Samples: 100
- Image height: 32 pixels
- Image width: 32 pixels
- Spectral bands: 200
- Data type: float32
- Format: HDF5

Original tensor shape:

(100, 32, 32, 200)

## 2. Data Loading

The hyperspectral dataset is loaded using the HDF5 (`h5py`) format.

The data loader reads:
- Hyperspectral data
- Corresponding labels

Labels represent:
- 0 = Healthy
- 1 = Diseased

## 3. Normalization

StandardScaler is used to normalize the spectral data.

Normalization is performed after reshaping the data into:

(samples × height × width, spectral bands)

This makes the spectral features suitable for PCA.

## 4. PCA

Principal Component Analysis (PCA) is applied to reduce the spectral dimensionality.

Original spectral bands:

200

PCA components:

20

After PCA, the tensor shape becomes:

(100, 32, 32, 20)

## 5. Batch Processing

The processed dataset is tested using batches of 10 samples.

Total samples:

100

Number of batches:

10

Each batch shape:

(10, 32, 32, 20)

## 6. Memory Testing

Memory usage was measured during batch processing.

Observed memory increase:

0.25 MB

This shows that batch processing can reduce unnecessary memory usage compared with loading the entire dataset repeatedly.

## 7. Final Pipeline

Hyperspectral Data
        ↓
Data Loading
        ↓
Normalization
        ↓
PCA (200 → 20 components)
        ↓
Batch Processing
        ↓
ML Model Input

## 8. Output

The processed dataset is saved as:

processed_hyperspectral.h5