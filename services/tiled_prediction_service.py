import torch

from inference.tiling import create_tiles
from inference.predict import predict_tiles


class TiledPredictionService:
    """
    Handles tiled raster prediction.

    The raster is divided into smaller tiles,
    each tile is passed through the model,
    and the results are aggregated.
    """

    def __init__(self, model):
        self.model = model

    def predict_raster(
        self,
        raster: torch.Tensor,
        tile_height: int = 8,
        tile_width: int = 8,
    ) -> dict:

        if not isinstance(
            raster,
            torch.Tensor,
        ):
            raise TypeError(
                "raster must be a torch.Tensor"
            )

        if raster.ndim != 4:
            raise ValueError(
                "Expected raster shape "
                "[C, D, H, W]. "
                f"Received "
                f"{tuple(raster.shape)}"
            )

        tiles = create_tiles(
            raster,
            tile_height=tile_height,
            tile_width=tile_width,
        )

        predictions = predict_tiles(
            self.model,
            tiles,
        )

        total_tiles = len(predictions)

        healthy_tiles = sum(
            1
            for result in predictions
            if result["class_id"] == 0
        )

        stressed_tiles = sum(
            1
            for result in predictions
            if result["class_id"] == 1
        )

        if total_tiles == 0:
            raise ValueError(
                "No tiles were generated."
            )

        average_probability = sum(
            result["probability"]
            for result in predictions
        ) / total_tiles

        stressed_ratio = (
            stressed_tiles / total_tiles
        )

        if stressed_tiles > healthy_tiles:
            overall_class_id = 1
            overall_class_name = (
                "Chemically Stressed"
            )
        else:
            overall_class_id = 0
            overall_class_name = "Healthy"

        return {
            "total_tiles": total_tiles,
            "healthy_tiles": healthy_tiles,
            "stressed_tiles": stressed_tiles,
            "stressed_ratio": stressed_ratio,
            "average_probability": (
                average_probability
            ),
            "overall_class_id": (
                overall_class_id
            ),
            "overall_class_name": (
                overall_class_name
            ),
            "tiles": predictions,
        }