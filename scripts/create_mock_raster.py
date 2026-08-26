import torch

from inference.tiling import create_tiles


def create_mock_raster():

    raster = torch.rand(
        1,
        8,
        32,
        32,
    )

    return raster


if __name__ == "__main__":

    raster = create_mock_raster()

    print("Mock hyperspectral raster created")
    print(f"Raster shape: {tuple(raster.shape)}")

    tiles = create_tiles(
        raster,
        tile_height=8,
        tile_width=8,
    )

    print(f"Number of tiles: {len(tiles)}")

    for index, item in enumerate(tiles):

        print(
            f"Tile {index + 1}: "
            f"shape={tuple(item['tile'].shape)}, "
            f"row={item['row']}, "
            f"col={item['col']}"
        )