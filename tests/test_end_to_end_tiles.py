import torch

from inference.tiling import create_tiles
from inference.predict import predict_tiles
from models.mock_model import MockModel


def test_end_to_end_tile_pipeline():

    # Create mock hyperspectral raster
    raster = torch.rand(
        1,
        8,
        16,
        16,
    )

    # Divide raster into tiles
    tiles = create_tiles(
        raster,
        tile_height=8,
        tile_width=8,
    )

    # Create mock model
    model = MockModel()

    # Run predictions
    results = predict_tiles(
        model,
        tiles,
    )

    # 16x16 raster with 8x8 tiles = 4 tiles
    assert len(results) == 4

    for result in results:

        assert "tile_id" in result
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