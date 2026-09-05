from fastapi.testclient import TestClient

from main import app


client = TestClient(app)


def test_predict_invalid_dimensions():

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

    assert (
        "Data length does not match"
        in response.json()["detail"]
    )

def test_predict_missing_data():

    payload = {
        "channels": 1,
        "depth": 8,
        "height": 8,
        "width": 8,
    }

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 422