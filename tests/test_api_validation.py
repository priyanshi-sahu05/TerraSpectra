from fastapi.testclient import TestClient

from main import app


client = TestClient(app)


def test_predict_rejects_wrong_data_length():

    payload = {
        "data": [0.0, 0.0],
        "channels": 1,
        "depth": 1,
        "height": 2,
        "width": 2,
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


def test_predict_rejects_invalid_dimensions():

    payload = {
        "data": [0.0],
        "channels": 0,
        "depth": 1,
        "height": 1,
        "width": 1,
    }

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 422

def test_predict_rejects_missing_data():

    payload = {
        "channels": 1,
        "depth": 1,
        "height": 1,
        "width": 1,
    }

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 422

def test_predict_tiles_rejects_wrong_data_length():

    payload = {
        "data": [0.0],
        "channels": 1,
        "depth": 8,
        "height": 16,
        "width": 16,
    }

    response = client.post(
        "/predict/tiles",
        json=payload,
    )

    assert response.status_code == 400