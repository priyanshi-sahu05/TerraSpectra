import h5py
import numpy as np

INPUT_FILE = "processed_hyperspectral.h5"
OUTPUT_FILE = "raster_tiles.h5"

DEFAULT_TILE_SIZE = 16


def generate_tiles(data, tile_size):
    """
    Split hyperspectral raster data into non-overlapping spatial tiles.

    Expected data shape:
        (samples, height, width, bands)
    """

    if data.ndim != 4:
        raise ValueError(
            f"Expected 4D hyperspectral data, got shape {data.shape}"
        )

    samples, height, width, bands = data.shape

    if tile_size <= 0:
        raise ValueError("Tile size must be greater than 0.")

    tiles = []

    for sample in range(samples):
        raster = data[sample]

        for row in range(0, height, tile_size):
            for col in range(0, width, tile_size):

                tile = raster[
                    row:min(row + tile_size, height),
                    col:min(col + tile_size, width),
                    :
                ]

                # Keep only complete tiles
                if tile.shape[0] == tile_size and tile.shape[1] == tile_size:
                    tiles.append(tile)

    if not tiles:
        raise ValueError("No complete tiles could be generated.")

    return np.asarray(tiles, dtype=np.float32)


def main():
    print("Loading hyperspectral raster...")

    with h5py.File(INPUT_FILE, "r") as f:
        if "data" in f:
            data = f["data"][:]
        elif "hyperspectral" in f:
            data = f["hyperspectral"][:]
        else:
            # Use the first dataset if the expected name is not present
            dataset_name = list(f.keys())[0]
            data = f[dataset_name][:]

    print("Input data shape:", data.shape)
    print("Tile size:", DEFAULT_TILE_SIZE)

    print("Generating raster tiles...")

    tiles = generate_tiles(data, DEFAULT_TILE_SIZE)

    print("Raster tiling completed!")
    print("Tiles shape:", tiles.shape)

    with h5py.File(OUTPUT_FILE, "w") as f:
        f.create_dataset("tiles", data=tiles)

    print("Raster tiles saved as:", OUTPUT_FILE)


if __name__ == "__main__":
    main()