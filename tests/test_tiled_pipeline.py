import torch

from inference.tiled_pipeline import (
    run_tiled_inference,
)
from models.mock_model import MockModel


def test_run_tiled_inference():

    raster = torch.rand(
        1,
        8,
        16,
        16,
    )

    model = MockModel()

    result = run_tiled_inference(
        model,
        raster,
        tile_height=8,
        tile_width=8,
    )

    assert "summary" in result
    assert "tiles" in result

    assert (
        result["summary"]["total_tiles"]
        == 4
    )

    assert len(
        result["tiles"]
    ) == 4

    for tile in result["tiles"]:

        assert "tile_id" in tile
        assert "row" in tile
        assert "col" in tile
        assert "class_id" in tile
        assert "probability" in tile