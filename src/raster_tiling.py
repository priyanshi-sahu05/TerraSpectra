import h5py
import numpy as np

INPUT_FILE = "processed_hyperspectral.h5"
OUTPUT_FILE = "raster_tiles.h5"

DEFAULT_TILE_SIZE = 16


def generate_tiles(data, tile_size):
    """
    Memory-efficient raster tiling.

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

    rows = height // tile_size
    cols = width // tile_size
    total_tiles = samples * rows * cols

    if total_tiles == 0:
        raise ValueError("No complete tiles could be generated.")

    # Pre-allocate output array to avoid list + np.array memory overhead
    tiles = np.empty(
        (total_tiles, tile_size, tile_size, bands),
        dtype=np.float32
    )

    index = 0

    for sample in range(samples):
        raster = data[sample]

        for row in range(rows):
            for col in range(cols):
                r_start = row * tile_size
                c_start = col * tile_size

                tiles[index] = raster[
                    r_start:r_start + tile_size,
                    c_start:c_start + tile_size,
                    :
                ]

                index += 1

    return tiles


def main():
    print("Loading hyperspectral raster...")

    with h5py.File(INPUT_FILE, "r") as f:
        if "data" in f:
            data = f["data"][:]
        elif "hyperspectral" in f:
            data = f["hyperspectral"][:]
        else:
            dataset_name = list(f.keys())[0]
            data = f[dataset_name][:]

    print("Input data shape:", data.shape)
    print("Tile size:", DEFAULT_TILE_SIZE)

    print("Generating raster tiles...")

    tiles = generate_tiles(data, DEFAULT_TILE_SIZE)

    print("Raster tiling completed!")
    print("Tiles shape:", tiles.shape)

    with h5py.File(OUTPUT_FILE, "w") as f:
        f.create_dataset(
            "tiles",
            data=tiles,
            compression="gzip"
        )

    print("Raster tiles saved as:", OUTPUT_FILE)


if __name__ == "__main__":
    main()