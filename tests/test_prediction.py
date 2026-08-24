import pytest
import torch

from inference.predict import predict_sample
from models.hybrid import CNNViTHybrid


def create_model():

    return CNNViTHybrid(
        cnn_channels=16,
        vit_channels=16,
        num_classes=2,
    )


def create_input():

    return torch.randn(
        1,
        1,
        8,
        8,
        8,
    )


def test_prediction_output_structure():

    model = create_model()
    input_tensor = create_input()

    result = predict_sample(
        model,
        input_tensor,
    )

    print("\nPrediction result:")
    print(result)

    assert isinstance(result, dict)

    assert set(result.keys()) == {
        "class_id",
        "class_name",
        "probability",
        "risk_score",
    }


def test_prediction_class_is_valid():

    model = create_model()
    input_tensor = create_input()

    result = predict_sample(
        model,
        input_tensor,
    )

    assert result["class_id"] in [0, 1]

    assert result["class_name"] in [
        "Healthy",
        "Chemically Stressed",
    ]


def test_probability_is_valid():

    model = create_model()
    input_tensor = create_input()

    result = predict_sample(
        model,
        input_tensor,
    )

    probability = result["probability"]

    assert 0.0 <= probability <= 1.0


def test_risk_score_is_valid():

    model = create_model()
    input_tensor = create_input()

    result = predict_sample(
        model,
        input_tensor,
    )

    risk_score = result["risk_score"]

    assert 0.0 <= risk_score <= 1.0


def test_prediction_with_invalid_dimensions():

    model = create_model()

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


def test_prediction_with_invalid_type():

    model = create_model()

    with pytest.raises(TypeError):

        predict_sample(
            model,
            "invalid input",
        )


def test_prediction_rejects_multiple_samples():

    model = create_model()

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


def test_model_is_in_evaluation_mode():

    model = create_model()

    model.train()

    input_tensor = create_input()

    predict_sample(
        model,
        input_tensor,
    )

    assert model.training is False

def test_multiple_independent_predictions():

    model = create_model()

    predictions = []

    for index in range(3):

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

        predictions.append(result)

    print("\nMultiple predictions:")

    for index, result in enumerate(predictions):

        print(
            f"Sample {index + 1}:",
            result,
        )

    assert len(predictions) == 3

    for result in predictions:

        assert result["class_id"] in [0, 1]

        assert (
            0.0
            <= result["probability"]
            <= 1.0
        )

        assert (
            0.0
            <= result["risk_score"]
            <= 1.0
        )