import torch

from inference.predict import predict_sample
from models.hybrid import CNNViTHybrid


def test_prediction_pipeline():

    model = CNNViTHybrid(
        cnn_channels=16,
        vit_channels=16,
        num_classes=2,
    )

    # One mock hyperspectral sample.
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

    print("Prediction result:")
    print(result)

    assert isinstance(result, dict)

    assert "class_id" in result
    assert "class_name" in result
    assert "probability" in result
    assert "risk_score" in result

    assert result["class_id"] in [0, 1]

    assert result["class_name"] in [
        "Healthy",
        "Chemically Stressed",
    ]

    assert 0.0 <= result["probability"] <= 1.0

    assert 0.0 <= result["risk_score"] <= 1.0

import pytest


def test_prediction_rejects_invalid_dimensions():

    model = CNNViTHybrid(
        cnn_channels=16,
        vit_channels=16,
        num_classes=2,
    )

    # Invalid: only 4 dimensions.
    invalid_input = torch.randn(
        1,
        8,
        8,
        8,
    )

    with pytest.raises(ValueError):
        predict_sample(
            model,
            invalid_input,
        )

def test_prediction_rejects_multiple_samples():

    model = CNNViTHybrid(
        cnn_channels=16,
        vit_channels=16,
        num_classes=2,
    )

    batch_input = torch.randn(
        2,
        1,
        8,
        8,
        8,
    )

    with pytest.raises(ValueError):
        predict_sample(
            model,
            batch_input,
        )