import numpy as np

from src.raster_tiling import generate_tiles, DEFAULT_TILE_SIZE


def main():
    print("Testing raster tiling...")

    # Dummy hyperspectral data
    # Shape: (samples, height, width, bands)
    data = np.random.rand(2, 32, 32, 20).astype(np.float32)

    print("Input shape:", data.shape)
    print("Tile size:", DEFAULT_TILE_SIZE)

    # Generate tiles
    tiles = generate_tiles(data, DEFAULT_TILE_SIZE)

    print("Output tiles shape:", tiles.shape)

    # Expected:
    # 2 samples × 2×2 spatial tiles = 8 tiles
    # Each tile = 16 × 16 × 20
    expected_shape = (8, 16, 16, 20)

    if tiles.shape == expected_shape:
        print("Raster tiling test PASSED!")
    else:
        print("Raster tiling test FAILED!")
        print("Expected:", expected_shape)
        print("Got:", tiles.shape)


if __name__ == "__main__":
    main()