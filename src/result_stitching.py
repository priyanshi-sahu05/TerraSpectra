import numpy as np


def stitch_predictions(predictions, image_height, image_width, tile_size):
    """
    Stitch tile-level predictions back into a full image map.

    Parameters:
        predictions: 1D array/list of predicted class labels
        image_height: original image height
        image_width: original image width
        tile_size: size of each square tile

    Returns:
        2D prediction map
    """

    predictions = np.asarray(predictions)

    tiles_y = image_height // tile_size
    tiles_x = image_width // tile_size

    expected_tiles = tiles_y * tiles_x

    if predictions.size != expected_tiles:
        raise ValueError(
            f"Expected {expected_tiles} predictions, "
            f"but got {predictions.size}"
        )

    result_map = np.zeros(
        (image_height, image_width),
        dtype=predictions.dtype
    )

    index = 0

    for row in range(tiles_y):
        for col in range(tiles_x):
            y_start = row * tile_size
            y_end = y_start + tile_size

            x_start = col * tile_size
            x_end = x_start + tile_size

            result_map[y_start:y_end, x_start:x_end] = predictions[index]
            index += 1

    return result_map


def main():
    print("Testing result stitching...")

    image_height = 32
    image_width = 32
    tile_size = 8

    # 16 tile predictions for a 32x32 image
    predictions = np.array([
        0, 1, 2, 3,
        1, 2, 3, 0,
        2, 3, 0, 1,
        3, 0, 1, 2
    ])

    result = stitch_predictions(
        predictions,
        image_height,
        image_width,
        tile_size
    )

    print("Prediction count:", len(predictions))
    print("Result map shape:", result.shape)
    print("Unique classes:", np.unique(result))

    assert result.shape == (32, 32)
    assert len(np.unique(result)) == 4

    print("Result stitching test PASSED!")


if __name__ == "__main__":
    main()