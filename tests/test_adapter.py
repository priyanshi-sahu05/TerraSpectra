import torch

from models.hybrid.adapter import CNNToViTAdapter


def test_adapter_output_shape():

    adapter = CNNToViTAdapter(
        input_channels=16,
        output_channels=16,
    )

    cnn_features = torch.randn(
        2,
        16,
        4,
        4,
        4,
    )

    adapted_features = adapter(cnn_features)

    print("CNN feature shape:", cnn_features.shape)
    print("Adapted feature shape:", adapted_features.shape)

    assert adapted_features.shape == (
        2,
        16,
        4,
        4,
        4,
    )

def test_adapter_changes_channels():

    adapter = CNNToViTAdapter(
        input_channels=32,
        output_channels=16,
    )

    cnn_features = torch.randn(
        2,
        32,
        4,
        4,
        4,
    )

    adapted_features = adapter(cnn_features)

    print(
        "Input channels:",
        cnn_features.shape[1],
    )

    print(
        "Output channels:",
        adapted_features.shape[1],
    )

    assert adapted_features.shape == (
        2,
        16,
        4,
        4,
        4,
    )