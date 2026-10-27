"""Regression tests for the issues found in the Fase 3 technical review.

Each test reproduces a defect observed against commit 549c30c and fails on it:
- an unhandled inference exception returned 500 but was never counted;
- arbitrary URLs became Prometheus label values (unbounded cardinality);
- whitespace/punctuation-only texts were accepted and "classified";
- a failed ONNX export had already overwritten the approved model.onnx.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from prometheus_client import REGISTRY

import src.api.main as api_main
from src.api.main import app

VALID_TEXT = "Patient presents with acute chest pain and elevated troponin levels."


def _count(endpoint: str, status_code: str) -> float:
    value = REGISTRY.get_sample_value(
        "vitorium_requests_total",
        {"method": "POST" if endpoint == "/predict" else "GET",
         "endpoint": endpoint, "status_code": status_code},
    )
    return value or 0.0


class _ExplodingBackend:
    name = "exploding"

    def predict_proba(self, texts):  # noqa: ANN001, ANN201 - test double
        raise RuntimeError("simulated inference failure")


def test_unhandled_inference_error_is_counted_once_as_500() -> None:
    with TestClient(app, raise_server_exceptions=False) as client:
        loaded = api_main._backend
        api_main._backend = _ExplodingBackend()
        before = _count("/predict", "500")
        try:
            response = client.post("/predict", json={"texto": VALID_TEXT})
        finally:
            api_main._backend = loaded
        after = _count("/predict", "500")
    assert response.status_code == 500
    assert after - before == 1.0


def test_unknown_paths_share_a_single_metric_label() -> None:
    with TestClient(app) as client:
        before = _count("unmatched", "404")
        for i in range(25):
            assert client.get(f"/scan-{i}/wp-admin").status_code == 404
        after = _count("unmatched", "404")
        body = client.get("/metrics").text
    assert after - before == 25.0
    assert "/scan-" not in body


@pytest.mark.parametrize("texto", [" " * 20, "!" * 20, "1234567890 ?!", "   ...   \n\t  "])
def test_texts_without_words_are_rejected(texto: str) -> None:
    with TestClient(app) as client:
        assert client.post("/predict", json={"texto": texto}).status_code == 422


def test_length_is_checked_after_stripping() -> None:
    with TestClient(app) as client:
        assert client.post("/predict", json={"texto": "   abc     "}).status_code == 422


def test_failed_onnx_export_keeps_the_approved_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("skl2onnx")
    import shutil

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import export_onnx

    models = Path(__file__).resolve().parents[1] / "models"
    if not (models / "model.joblib").exists() or not (models / "model.onnx").exists():
        pytest.skip("needs models/model.joblib and models/model.onnx")
    shutil.copy(models / "model.joblib", tmp_path / "model.joblib")
    shutil.copy(models / "model.onnx", tmp_path / "model.onnx")
    approved = hashlib.sha256((tmp_path / "model.onnx").read_bytes()).hexdigest()

    monkeypatch.setattr(export_onnx, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(export_onnx, "load_eval_texts", lambda: ([VALID_TEXT], "test"))
    monkeypatch.setattr(
        export_onnx, "check_parity",
        lambda *a, **k: {"rows_above_tolerance": 1, "label_agreement": 0.9},
    )
    monkeypatch.setattr(sys, "argv", ["export_onnx.py", "--skip-bench"])

    with pytest.raises(SystemExit):
        export_onnx.main()

    assert hashlib.sha256((tmp_path / "model.onnx").read_bytes()).hexdigest() == approved
    assert not (tmp_path / "model.onnx.candidate").exists()
