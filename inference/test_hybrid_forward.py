import torch

from models.hybrid import CNNViTHybrid


def main():

    print("\nTerraSpectra Hybrid Forward Pass")
    print("=" * 45)

    model = CNNViTHybrid(
        cnn_channels=16,
        vit_channels=16,
        num_classes=2,
    )

    model.eval()

    x = torch.randn(
        1,
        1,
        8,
        8,
        8,
    )

    print(
        "Input shape:",
        tuple(x.shape),
    )

    with torch.no_grad():

        cnn_features = (
            model.extract_cnn_features(x)
        )

        print(
            "CNN feature shape:",
            tuple(cnn_features.shape),
        )

        vit_features = (
            model.adapt_features(
                cnn_features
            )
        )

        print(
            "ViT feature shape:",
            tuple(vit_features.shape),
        )

        logits = model.vit(
            vit_features
        )

        print(
            "Logits shape:",
            tuple(logits.shape),
        )

        probabilities = torch.softmax(
            logits,
            dim=1,
        )

        print(
            "Probabilities:",
            probabilities.tolist(),
        )

    print("\nForward pass completed successfully.")


if __name__ == "__main__":
    main()