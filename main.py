from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch

from api.schemas import (
    PredictionRequest,
    PredictionResponse,
    TiledPredictionResponse,
)

from services.prediction_service import PredictionService

from inference.tiled_pipeline import (
    run_tiled_inference,
)
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
def predict(
    request: PredictionRequest,
):
    """
    Generate a prediction from validated input data.

    The current implementation accepts mock tensor data
    through the API. Real raster/tile input will be added later.
    """

    try:

        expected_size = (
            request.channels
            * request.depth
            * request.height
            * request.width
        )

        if len(request.data) != expected_size:

            raise HTTPException(
                status_code=400,
                detail=(
                    "Data length does not match "
                    "the provided tensor dimensions."
                ),
            )

        tensor = torch.tensor(
            request.data,
            dtype=torch.float32,
        )

        tensor = tensor.reshape(
            1,
            request.channels,
            request.depth,
            request.height,
            request.width,
        )

        result = prediction_service.predict(
            tensor
        )

        return result

    except HTTPException:
        raise

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(error)}",
        )

@app.post(
    "/predict/tiles",
    response_model=TiledPredictionResponse,
)
def predict_tiles(
    request: PredictionRequest,
):
    """
    Run tiled inference on a mock raster.

    The input tensor is divided into smaller tiles,
    predictions are generated for each tile,
    and the results are aggregated.
    """

    try:

        expected_size = (
            request.channels
            * request.depth
            * request.height
            * request.width
        )

        if len(request.data) != expected_size:

            raise HTTPException(
                status_code=400,
                detail=(
                    "Data length does not match "
                    "the provided tensor dimensions."
                ),
            )

        tensor = torch.tensor(
            request.data,
            dtype=torch.float32,
        )

        raster = tensor.reshape(
            request.channels,
            request.depth,
            request.height,
            request.width,
        )

        result = run_tiled_inference(
            prediction_service.model,
            raster,
            tile_height=8,
            tile_width=8,
        )

        return {
            **result["summary"],
            "tiles": result["tiles"],
        }

    except HTTPException:
        raise

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Tiled prediction failed: {str(error)}"
            ),
        )