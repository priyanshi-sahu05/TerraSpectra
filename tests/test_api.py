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

    response = client.post("/predict")

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