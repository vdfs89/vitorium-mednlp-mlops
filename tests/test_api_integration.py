"""Integration test for the /predict and /health routes."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api.main import app

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "model.joblib"


@pytest.mark.skipif(
    not MODEL_PATH.exists(), reason="Run `python -m src.training.train` first"
)
def test_health_reports_model_loaded() -> None:
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["model_loaded"] is True


@pytest.mark.skipif(
    not MODEL_PATH.exists(), reason="Run `python -m src.training.train` first"
)
def test_predict_returns_valid_urgency_tier() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={
                "texto": (
                    "Patient presents with acute chest pain radiating to the "
                    "left arm, elevated troponin levels, and ECG changes "
                    "consistent with myocardial infarction."
                )
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert body["urgencia"] in {"normal", "atencao", "urgente"}
        assert 0.0 <= body["confianca"] <= 1.0
        assert set(body["probabilidades"].keys()) == {"normal", "atencao", "urgente"}
        assert body["latencia_ms"] > 0


def test_predict_rejects_short_text() -> None:
    with TestClient(app) as client:
        response = client.post("/predict", json={"texto": "abc"})
        assert response.status_code == 422


def test_metrics_endpoint_exposes_prometheus_format() -> None:
    with TestClient(app) as client:
        client.get("/health")
        response = client.get("/metrics")
        assert response.status_code == 200
        assert b"vitorium_requests_total" in response.content
