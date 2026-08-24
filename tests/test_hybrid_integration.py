import torch

from models.hybrid import CNNViTHybrid


def test_cnn_feature_extraction():

    model = CNNViTHybrid(
        cnn_channels=16,
        vit_channels=16,
        num_classes=2,
    )

    x = torch.randn(
        2,
        1,
        8,
        8,
        8,
    )

    cnn_features = model.extract_cnn_features(x)

    print("\nInput shape:")
    print(x.shape)

    print("CNN feature shape:")
    print(cnn_features.shape)

    assert cnn_features.ndim == 5

    assert cnn_features.shape[0] == 2

    assert cnn_features.shape[1] == 16


def test_adapter_integration():

    model = CNNViTHybrid(
        cnn_channels=16,
        vit_channels=16,
        num_classes=2,
    )

    x = torch.randn(
        2,
        1,
        8,
        8,
        8,
    )

    cnn_features = model.extract_cnn_features(x)

    vit_features = model.adapt_features(
        cnn_features
    )

    print("\nCNN features:")
    print(cnn_features.shape)

    print("ViT features:")
    print(vit_features.shape)

    assert vit_features.ndim == 5

    assert vit_features.shape[0] == 2

    assert vit_features.shape[1] == 16


def test_complete_cnn_vit_integration():

    model = CNNViTHybrid(
        cnn_channels=16,
        vit_channels=16,
        num_classes=2,
    )

    x = torch.randn(
        2,
        1,
        8,
        8,
        8,
    )

    logits = model(x)

    print("\nFinal hybrid output:")
    print(logits.shape)

    assert logits.shape == (
        2,
        2,
    )