from inference.tiling import create_tiles
from inference.predict import predict_tiles
from inference.aggregation import (
    aggregate_tile_predictions,
)


def run_tiled_inference(
    model,
    raster,
    tile_height=8,
    tile_width=8,
):
    """
    Run complete tiled inference.

    Flow:

    Raster
       ↓
    Tiles
       ↓
    Model predictions
       ↓
    Aggregation
    """

    tiles = create_tiles(
        raster,
        tile_height=tile_height,
        tile_width=tile_width,
    )

    tile_predictions = predict_tiles(
        model,
        tiles,
    )

    summary = aggregate_tile_predictions(
        tile_predictions
    )

    return {
        "summary": summary,
        "tiles": tile_predictions,
    }