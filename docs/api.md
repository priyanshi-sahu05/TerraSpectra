# TerraSpectra Prediction API

## Base URL

http://127.0.0.1:8000

## Endpoints

### GET /

Checks whether the API service is running.

### GET /health

Checks API health and model availability.

### POST /predict

Generates a prediction for one hyperspectral sample.

### POST /predict/tiles

Runs prediction on multiple raster tiles and returns aggregated results.

---

# POST /predict/tiles

## Request

The API accepts flattened tensor data.

Example dimensions:

- channels: 1
- depth: 8
- height: 16
- width: 16

The number of values must be:

channels × depth × height × width

For the above example:

1 × 8 × 16 × 16 = 2048 values.

## Response

The response contains:

- total_tiles
- healthy_tiles
- stressed_tiles
- stressed_ratio
- average_probability
- overall_class_id
- overall_class_name
- tiles

Each tile contains:

- tile_id
- row
- col
- height
- width
- class_id
- class_name
- probability

## Class Labels

| Class ID | Class |
|---|---|
| 0 | Healthy |
| 1 | Chemically Stressed |

## Important

The current project uses mock input/model components for development.

The predictions are therefore demonstrations of the complete inference pipeline and are not final agricultural predictions.