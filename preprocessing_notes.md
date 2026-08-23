# TerraSpectra – Data Pipeline Documentation

## 1. Overview

The TerraSpectra preprocessing pipeline prepares hyperspectral crop data for machine learning and deep learning models.

The pipeline performs:
- Hyperspectral data loading
- Data cleaning and preprocessing
- PCA-based feature processing
- Spectral patch generation
- Dataset validation
- Edge-case checking

---

## 2. Input Dataset

The processed hyperspectral dataset is stored in:

`processed_hyperspectral.h5`

The input data has the following shape:

- Samples: 100
- Image height: 32
- Image width: 32
- Spectral bands: 20

Input data shape:

`(100, 32, 32, 20)`

Labels shape:

`(100,)`

---

## 3. Preprocessing Pipeline

The preprocessing workflow is:

```text
Hyperspectral Data
        |
        v
Data Loading
        |
        v
Data Cleaning / Normalization
        |
        v
PCA Feature Processing
        |
        v
Processed Hyperspectral Dataset
        |
        v
5 × 5 Spectral Patch Generation
        |
        v
Model-Ready Spectral Patches
        |
        v
Feature Validation
        |
        v
Edge-Case Validation