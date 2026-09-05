import torch

from models.mock_model import MockModel
from services.tiled_prediction_service import (
    TiledPredictionService,
)


def test_tiled_prediction_service():

    raster = torch.rand(
        1,
        8,
        16,
        16,
    )

    model = MockModel()

    service = TiledPredictionService(
        model
    )

    result = service.predict_raster(
        raster,
        tile_height=8,
        tile_width=8,
    )

    assert result["total_tiles"] == 4

    assert (
        result["healthy_tiles"]
        + result["stressed_tiles"]
        == 4
    )

    assert (
        0.0
        <= result["stressed_ratio"]
        <= 1.0
    )

    assert (
        0.0
        <= result["average_probability"]
        <= 1.0
    )

    assert result["overall_class_id"] in [
        0,
        1,
    ]

    assert result["overall_class_name"] in [
        "Healthy",
        "Chemically Stressed",
    ]

    assert len(result["tiles"]) == 4