
from fastapi.testclient import TestClient

from backend.app.main import app


def test_health():
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_predict_intent():
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/predict-intent",
            json={
                "message": "Please give me my money back"
            },
        )

    assert response.status_code == 200

    data = response.json()

    assert data["intent"] == "refund"
    assert 0 <= data["probability"] <= 1
    assert len(data["probabilities"]) == 4


def test_empty_message():
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/predict-intent",
            json={"message": ""},
        )

    assert response.status_code == 422
