"""Smoke test: the app imports and the model artifact loads."""

from fastapi.testclient import TestClient

from src.api.main import app


def test_app_starts() -> None:
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
