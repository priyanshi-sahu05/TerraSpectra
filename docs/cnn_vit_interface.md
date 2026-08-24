# CNN → ViT Interface

## Project

TerraSpectra — Hyperspectral Crop Disease Forecasting

## Member

Member 3 — Vision Transformer + FastAPI

## Current Development Status

Member 2's 3D-CNN has not yet been implemented.

Therefore, Member 3 development currently uses a temporary
mock CNN feature extractor.

The mock implementation is only for development and testing.

## Mock CNN Input

```text
[B, 1, D, H, W]

## Standalone ViT Development

A standalone Vision Transformer has been implemented using
controlled mock CNN feature tensors.

### Mock CNN Feature

```text
[B, 16, D, H, W]