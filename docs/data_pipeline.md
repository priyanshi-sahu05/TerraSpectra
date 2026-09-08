# TerraSpectra - Data Pipeline Documentation

## 1. Overview

The TerraSpectra data pipeline prepares hyperspectral crop data for machine learning and deep learning models.

The pipeline includes dataset loading, data validation, normalization, dimensionality reduction using PCA, spectral patch generation, and raster tiling.

## 2. Dataset

The project uses hyperspectral image data containing multiple spectral bands.

Initial mock hyperspectral dataset shape:

- Samples: 100
- Height: 32
- Width: 32
- Spectral bands: 200

Initial shape:

`(100, 32, 32, 200)`

## 3. Data Validation

The input datasets are checked for:

- Dataset dimensions
- Data types
- Sample shapes
- Missing or invalid values
- Infinite values

The validation script tests the processed hyperspectral dataset and generated spectral patches.

## 4. Normalization

Normalization is applied to the hyperspectral data so that feature values remain within a suitable numerical range.

This helps improve model training stability and convergence.

## 5. PCA Dimensionality Reduction

Principal Component Analysis (PCA) is used to reduce the number of spectral bands while preserving important information.

The original spectral dimension is reduced from:

`200 bands -> 20 principal components`

After PCA, the processed dataset has the shape:

`(100, 32, 32, 20)`

This reduces computational cost and makes the data more suitable for deep learning models.

## 6. Spectral Patch Generation

Small spatial-spectral patches are generated from the processed hyperspectral data.

The generated spectral patches have the shape:

`(78400, 5, 5, 20)`

Where:

- 78400 = number of generated patches
- 5 x 5 = spatial patch size
- 20 = PCA spectral components

These patches are used as model-ready input for CNN and other machine learning models.

## 7. Raster Tiling

Raster tiling divides large hyperspectral images into smaller tiles.

The project includes a reusable raster tiling function in:

`src/raster_tiling.py`

The default tile size is:

`16 x 16`

Tiling helps process large raster images efficiently and supports batch model inference.

## 8. Model Input

The processed spectral patches can be converted into the required tensor format for the 3D-CNN model.

The model processes spatial and spectral information together.

Typical CNN input format:

`(N, 1, spectral_bands, height, width)`

Example:

`(N, 1, 20, 5, 5)`

## 9. Tools and Libraries

The data pipeline uses:

- Python
- NumPy
- H5Py
- Scikit-learn
- Rasterio/GDAL
- PyTorch

## 10. Data Pipeline Flow

The overall pipeline is:

Raw Hyperspectral Data

↓

Data Validation

↓

Normalization

↓

PCA Dimensionality Reduction

↓

Processed Hyperspectral Dataset

↓

Spectral Patch Generation

↓

Raster Tiling

↓

Model-ready Tensor

↓

3D-CNN / ML Model

## 11. Output Files

Important generated files include:

- `processed_hyperspectral.h5`
- `spectral_patches.h5`

Large generated datasets are kept locally and are excluded from Git tracking when required.

## 12. Conclusion

The data pipeline converts hyperspectral data into validated, normalized, dimensionally reduced, and model-ready data.

PCA reduces the spectral dimensions from 200 to 20 components, while spectral patches and raster tiling make the data suitable for efficient deep learning and inference.