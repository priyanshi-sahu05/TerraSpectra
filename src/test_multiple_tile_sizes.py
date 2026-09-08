import numpy as np
from src.raster_tiling import generate_tiles


def main():
    print("Testing multiple tile sizes...")

    # Dummy hyperspectral image: H, W, Bands
    image = np.random.rand(2, 32, 32, 20).astype(np.float32)

    tile_sizes = [8, 16, 32]

    for size in tile_sizes:
        tiles = generate_tiles(image, tile_size=size)

        print(f"Tile size: {size}")
        print(f"Tiles shape: {tiles.shape}")
        print("-" * 30)

    print("Multiple tile size test PASSED!")


if __name__ == "__main__":
    main()