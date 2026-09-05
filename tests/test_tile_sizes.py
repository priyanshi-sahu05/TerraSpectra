import torch

from inference.tiling import create_tiles


def test_16x16_raster():

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

    assert len(tiles) == 4


def test_32x32_raster():

    raster = torch.rand(
        1,
        8,
        32,
        32,
    )

    tiles = create_tiles(
        raster,
        tile_height=8,
        tile_width=8,
    )

    assert len(tiles) == 16