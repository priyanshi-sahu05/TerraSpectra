import torch

from services.prediction_service import PredictionService


def test_prediction_service_creation():

    service = PredictionService()

    assert service.model is not None

    assert service.model.training is False


def test_prediction_service_prediction():

    service = PredictionService()

    input_tensor = torch.randn(
        1,
        1,
        8,
        8,
        8,
    )

    result = service.predict(
        input_tensor
    )

    print("\nService prediction:")
    print(result)

    assert isinstance(
        result,
        dict,
    )

    assert "class_id" in result
    assert "class_name" in result
    assert "probability" in result
    assert "risk_score" in result


def test_prediction_service_probability():

    service = PredictionService()

    input_tensor = torch.randn(
        1,
        1,
        8,
        8,
        8,
    )

    result = service.predict(
        input_tensor
    )

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

def test_prediction_service_batch():

    service = PredictionService()

    input_tensor = torch.randn(
        3,
        1,
        8,
        8,
        8,
    )

    results = service.predict_batch(
        input_tensor
    )

    print("\nBatch predictions:")

    for result in results:
        print(result)

    assert len(results) == 3

    for index, result in enumerate(
        results
    ):

        assert (
            result["sample_index"]
            == index
        )

        assert result["class_id"] in [
            0,
            1,
        ]

        assert (
            0.0
            <= result["risk_score"]
            <= 1.0
        )