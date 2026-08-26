import torch
import pytest

from inference.tiling import create_tiles


def test_create_tiles():

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

    for item in tiles:

        assert tuple(
            item["tile"].shape
        ) == (1, 8, 8, 8)


def test_tile_positions():

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

    positions = [
        (item["row"], item["col"])
        for item in tiles
    ]

    expected_positions = [
        (0, 0),
        (0, 8),
        (8, 0),
        (8, 8),
    ]

    assert positions == expected_positions


def test_edge_tiles():

    raster = torch.rand(
        1,
        8,
        10,
        10,
    )

    tiles = create_tiles(
        raster,
        tile_height=6,
        tile_width=6,
    )

    assert len(tiles) == 4

    assert tuple(
        tiles[-1]["tile"].shape
    ) == (1, 8, 4, 4)


def test_invalid_raster_shape():

    raster = torch.rand(
        8,
        32,
        32,
    )

    with pytest.raises(ValueError):

        create_tiles(
            raster,
            tile_height=8,
            tile_width=8,
        )