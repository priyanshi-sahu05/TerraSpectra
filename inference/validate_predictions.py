import torch

from inference.predict import predict_sample
from models.hybrid import CNNViTHybrid

torch.manual_seed(42)

def main():

    model = CNNViTHybrid(
        cnn_channels=16,
        vit_channels=16,
        num_classes=2,
    )

    print("\nTerraSpectra Prediction Validation")
    print("=" * 40)

    for index in range(5):

        input_tensor = torch.randn(
            1,
            1,
            8,
            8,
            8,
        )

        result = predict_sample(
            model,
            input_tensor,
        )

        print(
            f"\nSample {index + 1}"
        )

        print(
            "Class:",
            result["class_name"],
        )

        print(
            "Probability:",
            result["probability"],
        )

        print(
            "Risk score:",
            result["risk_score"],
        )


if __name__ == "__main__":
    main()