# CNN → ViT Interface

## Project

TerraSpectra — Hyperspectral Crop Disease Forecasting

## Member

Member 3 — Vision Transformer + FastAPI

## Prediction Validation

Prediction validation checks the output of the hybrid CNN + ViT
inference pipeline.

The following conditions are validated:

- Input tensor dimensions
- Input tensor type
- Valid class ID
- Valid class name
- Probability range
- Risk score range
- Probability distribution
- Evaluation mode
- Multiple independent predictions

### Current Mock Input

```text
[B, 1, 8, 8, 8]