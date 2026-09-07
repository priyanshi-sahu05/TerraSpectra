import torch
import h5py
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

from models.hybrid_model import HybridCNNViT


def main():
    print("Validating final Hybrid CNN + ViT model...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Load spectral patches
    with h5py.File("spectral_patches.h5", "r") as f:
        X = f["patches"][:]
        y = f["labels"][:]

    print("Dataset shape:", X.shape)
    print("Labels shape:", y.shape)

    # Convert to tensors
    X = torch.tensor(X, dtype=torch.float32).permute(0, 3, 1, 2)
    y = torch.tensor(y, dtype=torch.long)

    # Model input: (N, 1, 20, 5, 5)
    X = X.unsqueeze(1)

    # Same train/test split used during training
    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("Validation samples:", len(X_test))

    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(
        test_dataset,
        batch_size=8,
        shuffle=False
    )

    # Create model
    model = HybridCNNViT(num_classes=4).to(device)

    model.load_state_dict(torch.load("models/final_hybrid_model.pth", map_location=device))
    model.eval()

    model.eval()

    correct = 0
    total = 0
    all_predictions = []
    all_labels = []

    with torch.inference_mode():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            output = model(batch_x)
            predictions = torch.argmax(output, dim=1)

            correct += (predictions == batch_y).sum().item()
            total += batch_y.size(0)

            all_predictions.extend(predictions.cpu().tolist())
            all_labels.extend(batch_y.cpu().tolist())

    accuracy = correct / total

    print("\n--- Final Model Validation ---")
    print("Correct predictions:", correct)
    print("Total predictions:", total)
    print(f"Validation Accuracy: {accuracy * 100:.2f}%")

    print("First 20 predictions:", all_predictions[:20])
    print("First 20 actual labels:", all_labels[:20])

    print("\nPrediction count:", len(all_predictions))

    if len(all_predictions) == len(all_labels):
        print("Prediction validation PASSED!")
    else:
        print("Prediction validation FAILED!")


if __name__ == "__main__":
    main()