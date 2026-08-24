import torch

from inference.predict import predict_sample
from models.hybrid import CNNViTHybrid


def main():

    model = CNNViTHybrid(
        cnn_channels=16,
        vit_channels=16,
        num_classes=2,
    )

    mock_input = torch.randn(
        1,
        1,
        8,
        8,
        8,
    )

    result = predict_sample(
        model,
        mock_input,
    )

    print("\nTerraSpectra Prediction")
    print("------------------------")
    print("Class ID:", result["class_id"])
    print("Class:", result["class_name"])
    print("Probability:", result["probability"])
    print("Risk Score:", result["risk_score"])


if __name__ == "__main__":
    main()