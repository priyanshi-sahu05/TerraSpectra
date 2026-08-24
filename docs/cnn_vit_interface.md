# CNN → ViT Interface

## Project

TerraSpectra — Hyperspectral Crop Disease Forecasting

## Member

Member 3 — Vision Transformer + FastAPI

## CNN → ViT Adapter

A dedicated adapter layer is used between the CNN and ViT.

The adapter currently uses a `1×1×1` 3D convolution to project
CNN feature channels into the feature dimension expected by
the ViT.

Example:

```text
CNN feature
[B, 32, D, H, W]
        ↓
1×1×1 projection
        ↓
[B, 16, D, H, W]
        ↓
ViT