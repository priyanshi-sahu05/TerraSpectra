from fastapi.testclient import TestClient

from main import app


client = TestClient(app)


def test_root_endpoint():

    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert data["project"] == "TerraSpectra"

    assert data["status"] == "running"


def test_health_endpoint():

    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"

    assert data["model_loaded"] is True


def test_predict_endpoint():

    total_values = 1 * 8 * 8 * 8

    payload = {
        "data": [0.1 for _ in range(total_values)],
        "channels": 1,
        "depth": 8,
        "height": 8,
        "width": 8,
    }

    response = client.post(
        "/predict",
        json=payload,
    )

    print("\nAPI response:")
    print(response.json())

    assert response.status_code == 200

    data = response.json()

    assert "class_id" in data
    assert "class_name" in data
    assert "probability" in data
    assert "risk_score" in data

    assert data["class_id"] in [0, 1]
    assert 0.0 <= data["probability"] <= 1.0
    assert 0.0 <= data["risk_score"] <= 1.0

def test_predict_invalid_data_length():

    payload = {
        "data": [0.1, 0.2, 0.3],
        "channels": 1,
        "depth": 8,
        "height": 8,
        "width": 8,
    }

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 400

    data = response.json()

    assert (
        "Data length does not match"
        in data["detail"]
    )

def test_predict_valid_input():

    total_values = (
        1 * 8 * 8 * 8
    )

    payload = {
        "data": [
            0.1
            for _ in range(total_values)
        ],
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