from fastapi.testclient import TestClient

from main import app


client = TestClient(app)


def test_health():

    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"

    assert data["service"] == (
        "terraspectra-prediction-api"
    )

    assert "model_loaded" in data

def test_root():

    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert data["project"] == "TerraSpectra"

    assert data["service"] == "Prediction API"

    assert data["status"] == "running"