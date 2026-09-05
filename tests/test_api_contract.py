from fastapi.testclient import TestClient

from main import app


client = TestClient(app)


def test_health_response():

    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert "service" in data
    assert "model_loaded" in data


def test_predict_response_contract():

    payload = {
        "data": [0.0] * 512,
        "channels": 1,
        "depth": 8,
        "height": 8,
        "width": 8,
    }

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 200

    data = response.json()

    assert "class_id" in data
    assert "class_name" in data
    assert "probability" in data
    assert "risk_score" in data

    assert data["class_id"] in [0, 1]

    assert 0.0 <= data["probability"] <= 1.0

    assert 0.0 <= data["risk_score"] <= 1.0