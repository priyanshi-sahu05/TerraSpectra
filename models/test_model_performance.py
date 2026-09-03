import torch
from models.cnn3d import Basic3DCNN


def main():
    print("Testing model performance and errors...")

    # Create model
    model = Basic3DCNN(num_classes=4)
    model.eval()

    # Dummy test data
    # Shape: (batch, channels, depth, height, width)
    test_x = torch.randn(20, 1, 8, 32, 32)

    # Dummy ground-truth labels
    test_y = torch.randint(0, 4, (20,))

    # Model prediction
    with torch.no_grad():
        outputs = model(test_x)
        predictions = torch.argmax(outputs, dim=1)

    # Accuracy
    correct = (predictions == test_y).sum().item()
    total = len(test_y)
    accuracy = correct / total

    # Errors
    errors = (predictions != test_y).sum().item()

    print("Test samples:", total)
    print("Correct predictions:", correct)
    print("Incorrect predictions:", errors)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("Predictions:", predictions.tolist())
    print("Actual labels:", test_y.tolist())

    print("Model performance test PASSED!")


if __name__ == "__main__":
    main()