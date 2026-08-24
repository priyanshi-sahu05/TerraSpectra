# CNN → ViT Interface

## Project

TerraSpectra — Hyperspectral Crop Disease Forecasting

## Member

Member 3 — Vision Transformer + FastAPI

## Prediction Pipeline

The hybrid model is exposed through a reusable prediction
function.

The inference process is:

```text
Input Tensor
    ↓
Hybrid CNN + ViT
    ↓
Logits
    ↓
Softmax
    ↓
Class Probability
    ↓
Predicted Class
    ↓
Risk Score
