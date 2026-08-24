from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch

from api.schemas import PredictionResponse
from services.prediction_service import PredictionService


app = FastAPI(
    title="TerraSpectra Prediction API",
    description=(
        "FastAPI service for TerraSpectra "
        "hyperspectral crop disease prediction."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

prediction_service = PredictionService()


@app.get("/")
def root():
    return {
        "project": "TerraSpectra",
        "service": "Prediction API",
        "status": "running",
    }


@app.get("/health")
def health_check():

    return {
        "status": "healthy",
        "service": "terraspectra-prediction-api",
        "model_loaded": (
            prediction_service.model
            is not None
        ),
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict():
    """
    Temporary development prediction endpoint.

    Currently uses a mock hyperspectral tensor because
    the real raster input pipeline is not connected yet.
    """

    try:

        mock_input = torch.randn(
            1,
            1,
            8,
            8,
            8,
        )

        result = prediction_service.predict(
            mock_input
        )

        return result

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(error)}",
        )