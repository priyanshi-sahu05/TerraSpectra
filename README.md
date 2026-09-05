{
  "total_tiles": 4,
  "healthy_tiles": 3,
  "stressed_tiles": 1,
  "stressed_ratio": 0.25,
  "average_probability": 0.72,
  "overall_class_id": 1,
  "overall_class_name": "Chemically Stressed",
  "tiles": [
    {
      "tile_id": 0,
      "row": 0,
      "col": 0,
      "height": 8,
      "width": 8,
      "class_id": 0,
      "class_name": "Healthy",
      "probability": 0.71
    }
  ]
}## Member 3 — ViT + FastAPI

TerraSpectra uses a hybrid CNN + Vision Transformer architecture
for hyperspectral crop stress prediction.

### Prediction Flow

Hyperspectral Tensor
        ↓
3D CNN
        ↓
Feature Representation
        ↓
Vision Transformer
        ↓
Classification
        ↓
Prediction Service
        ↓
FastAPI

### API Endpoints

GET /

Returns the API service status.

GET /health

Checks whether the prediction service is healthy.

POST /predict

Performs prediction on a single input tensor.

POST /predict/tiles

Splits mock raster data into tiles, performs prediction on each
tile, and returns aggregated results.

### Tiled Prediction

Large raster data is divided into smaller tiles.

Each tile is processed independently and produces:

- tile ID
- row
- column
- tile dimensions
- predicted class
- class name
- probability

The API also calculates:

- total tiles
- healthy tiles
- stressed tiles
- stressed ratio
- average probability
- overall prediction

### Current Limitation

The current implementation uses mock hyperspectral data and
a mock/untrained model for development and API testing.

The architecture is designed so that the real trained CNN/model
weights can be connected later.