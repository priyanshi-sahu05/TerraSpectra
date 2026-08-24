# CNN → ViT Interface

## Project

TerraSpectra — Hyperspectral Crop Disease Forecasting

## Member

Member 3 — Vision Transformer + FastAPI

## Forward-Pass Validation

The hybrid model was tested using mock hyperspectral tensors.

The forward path is:

```text
Input Tensor
    ↓
3D-CNN
    ↓
CNN Feature Tensor
    ↓
CNN → ViT Adapter
    ↓
ViT Feature Representation
    ↓
Transformer Encoder
    ↓
Classification Head
    ↓
Logits
    ↓
Softmax
    ↓
Class Probabilities