## FastAPI Implementation

The first FastAPI version has been implemented.

### Available Endpoints

#### GET /

Returns basic API information.

#### GET /health

Checks whether the API is running and whether the
prediction model has been loaded.

#### POST /predict

Runs a prediction using the current mock hyperspectral
input.

### Current Development Flow

```text
POST /predict
     ↓
Mock Tensor
     ↓
PredictionService
     ↓
CNN + ViT
     ↓
Prediction
     ↓
JSON