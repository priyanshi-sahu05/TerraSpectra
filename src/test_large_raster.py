import numpy as np
from src.raster_tiling import generate_tiles


def main():
    print("Testing large raster tiling...")

    # Simulated large hyperspectral raster
    height = 256
    width = 256
    bands = 20
    tile_size = 32

    raster = np.random.rand(1, height, width, bands).astype(np.float32)

    print("Raster shape:", raster.shape)
    print("Tile size:", tile_size)

    tiles = generate_tiles(raster, tile_size)

    print("Generated tiles:", tiles.shape)
    print("Total tiles:", len(tiles))

    assert tiles.shape[1:] == (tile_size, tile_size, bands)

    print("Large raster stress test PASSED!")


if __name__ == "__main__":
    main()