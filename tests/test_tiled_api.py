from fastapi.testclient import TestClient

from main import app


client = TestClient(app)


def test_tiled_api():

    response = client.post(
        "/predict/tiles",
        json={
            "data": [0.0],
            "channels": 1,
            "depth": 1,
            "height": 1,
            "width": 1,
        },
    )

    assert response.status_code == 200

    result = response.json()

    assert "total_tiles" in result
    assert "healthy_tiles" in result
    assert "stressed_tiles" in result
    assert "stressed_ratio" in result
    assert "average_probability" in result
    assert "overall_class_id" in result
    assert "overall_class_name" in result
    assert "tiles" in result

    assert result["total_tiles"] == 1

    assert len(result["tiles"]) == 1


def test_tiled_api_rejects_wrong_data_size():

    response = client.post(
        "/predict/tiles",
        json={
            "data": [0.0],
            "channels": 1,
            "depth": 1,
            "height": 2,
            "width": 2,
        },
    )

    assert response.status_code == 400