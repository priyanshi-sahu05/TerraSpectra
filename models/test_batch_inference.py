import torch
from torch.utils.data import DataLoader, TensorDataset

from models.cnn3d import Basic3DCNN


def main():
    print("Testing batch inference...")

    # Dummy hyperspectral tiles
    # Shape: (samples, channels, depth, height, width)
    x = torch.randn(8, 1, 8, 32, 32)

    # Dummy labels
    y = torch.randint(0, 4, (8,))

    dataset = TensorDataset(x, y)

    # Batch inference
    batch_size = 4
    test_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False
    )

    # Create CNN model
    model = Basic3DCNN(num_classes=4)
    model.eval()

    all_predictions = []

    with torch.no_grad():
        for batch_x, _ in test_loader:

            output = model(batch_x)

            predictions = torch.argmax(output, dim=1)

            all_predictions.extend(predictions.tolist())

            print(
                f"Batch input shape: {batch_x.shape}, "
                f"Output shape: {output.shape}"
            )

    print("Total predictions:", len(all_predictions))
    print("Predictions:", all_predictions)
    print("Batch inference test PASSED!")


if __name__ == "__main__":
    main()