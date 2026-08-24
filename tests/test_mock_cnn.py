import torch

from models.mock_cnn import MockCNN3D


def test_mock_cnn_feature_shape():
    model = MockCNN3D()

    # Development-only mock hyperspectral input.
    x = torch.randn(
        2, 1, 16, 16, 16
    )

    features = model.forward_features(x)

    print("Input shape:", x.shape)
    print("Feature shape:", features.shape)

    assert features.shape == (2, 16, 16, 16, 16)