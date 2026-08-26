def aggregate_tile_predictions(
    tile_predictions,
):
    """
    Aggregate predictions from multiple tiles.

    Returns a summary of the raster-level prediction.
    """

    if not tile_predictions:
        raise ValueError(
            "tile_predictions cannot be empty"
        )

    total_tiles = len(
        tile_predictions
    )

    stressed_tiles = sum(
        1
        for item in tile_predictions
        if item["class_id"] == 1
    )

    healthy_tiles = sum(
        1
        for item in tile_predictions
        if item["class_id"] == 0
    )

    average_probability = sum(
        item["probability"]
        for item in tile_predictions
    ) / total_tiles

    stressed_ratio = (
        stressed_tiles / total_tiles
    )

    if stressed_ratio >= 0.5:
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
        "average_probability": average_probability,
        "overall_class_id": overall_class_id,
        "overall_class_name": overall_class_name,
    }