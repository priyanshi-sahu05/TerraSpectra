import torch

from models.hybrid import CNNViTHybrid


def main():

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

    print("\nTerraSpectra Hybrid Model")
    print("=" * 40)

    print(
        "Input:",
        tuple(x.shape),
    )

    with torch.no_grad():

        cnn_features = model.extract_cnn_features(x)

        print(
            "CNN features:",
            tuple(cnn_features.shape),
        )

        vit_features = model.adapt_features(
            cnn_features
        )

        print(
            "ViT features:",
            tuple(vit_features.shape),
        )

        logits = model.vit(
            vit_features
        )

        print(
            "Final logits:",
            tuple(logits.shape),
        )

    print("\nTensor flow:")
    print(
        f"{tuple(x.shape)}"
        " → "
        f"{tuple(cnn_features.shape)}"
        " → "
        f"{tuple(vit_features.shape)}"
        " → "
        f"{tuple(logits.shape)}"
    )


if __name__ == "__main__":
    main()