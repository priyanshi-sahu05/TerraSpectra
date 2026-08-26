import torch

from inference.tiling import create_tiles
from inference.predict import predict_tiles
from models.mock_model import MockModel


def test_predict_tiles():

    raster = torch.rand(
        1,
        8,
        16,
        16,
    )

    tiles = create_tiles(
        raster,
        tile_height=8,
        tile_width=8,
    )

    model = MockModel()

    results = predict_tiles(
        model,
        tiles,
    )

    assert len(results) == 4

    for index, result in enumerate(results):

        assert result["tile_id"] == index

        assert "row" in result
        assert "col" in result
        assert "height" in result
        assert "width" in result

        assert "class_id" in result
        assert "class_name" in result
        assert "probability" in result

        assert result["class_id"] in [0, 1]

        assert result["class_name"] in [
            "Healthy",
            "Chemically Stressed",
        ]

        assert (
            0.0
            <= result["probability"]
            <= 1.0
        )