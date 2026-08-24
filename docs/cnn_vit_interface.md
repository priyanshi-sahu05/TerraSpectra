# CNN → ViT Interface

## Project

TerraSpectra — Hyperspectral Crop Disease Forecasting

## Member

Member 3 — Vision Transformer + FastAPI

## Hybrid Integration

The current development model exposes three stages:

1. CNN feature extraction
2. CNN-to-ViT feature adaptation
3. Vision Transformer classification

### Tensor Flow

```text
Input
[B, C, D, H, W]
      ↓
3D-CNN
      ↓
CNN Feature Tensor
[B, C1, D1, H1, W1]
      ↓
CNN → ViT Adapter
      ↓
ViT Feature Tensor
[B, C2, D1, H1, W1]
      ↓
Vision Transformer
      ↓
Classification Logits
[B, num_classes]