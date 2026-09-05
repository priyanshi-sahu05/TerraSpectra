import torch

from inference.tiling import create_tiles
from inference.predict import predict_tiles
from models.mock_model import MockModel


def main():

    raster = torch.rand(
        1,
        8,
        16,
        16,
    )

    print("Raster shape:")
    print(raster.shape)

    tiles = create_tiles(
        raster,
        tile_height=8,
        tile_width=8,
    )

    print("\nNumber of tiles:")
    print(len(tiles))

    model = MockModel()

    results = predict_tiles(
        model,
        tiles,
    )

    print("\nTile predictions:")

    for result in results:
        print(result)


if __name__ == "__main__":
    main()