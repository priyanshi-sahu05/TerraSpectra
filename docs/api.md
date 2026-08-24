# TerraSpectra Prediction API

## Purpose

The Prediction API exposes the hybrid CNN + Vision Transformer
model to external applications such as the React GIS dashboard.

## Planned Flow

```text
React Dashboard
      ↓
POST /predict
      ↓
FastAPI
      ↓
Prediction Service
      ↓
CNN + ViT
      ↓
Prediction
      ↓
JSON Response