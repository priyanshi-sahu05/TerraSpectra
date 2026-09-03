import torch
from models.cnn3d import Basic3DCNN


def main():
    print("Testing optimized inference...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    model = Basic3DCNN(num_classes=4).to(device)
    model.eval()

    # Batch of tiles
    tiles = torch.randn(16, 1, 8, 32, 32).to(device)

    # Optimized inference: no gradient calculation
    with torch.inference_mode():
        outputs = model(tiles)
        predictions = torch.argmax(outputs, dim=1)

    print("Batch size:", tiles.size(0))
    print("Input shape:", tiles.shape)
    print("Output shape:", outputs.shape)
    print("Predictions:", predictions.tolist())

    print("Optimized inference test PASSED!")


if __name__ == "__main__":
    main()
    