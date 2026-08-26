import torch


def create_tiles(
    raster: torch.Tensor,
    tile_height: int,
    tile_width: int,
):
    """
    Divide a raster tensor into smaller spatial tiles.

    Expected raster shape:

        [C, D, H, W]

    Returns a list of dictionaries containing
    the tile tensor and its spatial position.
    """

    if raster.ndim != 4:
        raise ValueError(
            "Raster must have shape [C, D, H, W]."
        )

    if tile_height <= 0 or tile_width <= 0:
        raise ValueError(
            "Tile height and width must be greater than zero."
        )

    channels, depth, height, width = raster.shape

    tiles = []

    for row in range(0, height, tile_height):

        for col in range(0, width, tile_width):

            end_row = min(
                row + tile_height,
                height,
            )

            end_col = min(
                col + tile_width,
                width,
            )

            tile = raster[
                :,
                :,
                row:end_row,
                col:end_col,
            ]

            tiles.append(
                {
                    "tile": tile,
                    "row": row,
                    "col": col,
                    "height": end_row - row,
                    "width": end_col - col,
                }
            )

    return tiles