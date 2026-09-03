import torch
from models.cnn3d import Basic3DCNN


def main():
    print("Validating predictions from tiles...")

    # Create model
    model = Basic3DCNN(num_classes=4)
    model.eval()

    # Dummy tile data
    # Shape: (batch, channels, depth, height, width)
    tiles = torch.randn(8, 1, 8, 32, 32)

    with torch.no_grad():
        outputs = model(tiles)
        predictions = torch.argmax(outputs, dim=1)

    print("Tiles shape:", tiles.shape)
    print("Output shape:", outputs.shape)
    print("Predictions:", predictions.tolist())

    # Validation checks
    assert outputs.shape == (8, 4), "Unexpected output shape"
    assert predictions.shape == (8,), "Unexpected prediction shape"
    assert torch.all((predictions >= 0) & (predictions < 4)), \
        "Invalid prediction class"

    print("Tile prediction validation PASSED!")


if __name__ == "__main__":
    main()