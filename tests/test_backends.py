"""Tests for the selectable inference backend (sklearn vs. ONNX Runtime).

The ONNX model is exported *inside the test* from whatever ``model.joblib`` is
on disk, never taken from the committed ``model.onnx``: CI retrains the
joblib from scratch, so a committed ONNX file could legitimately belong to a
different fit. This way the parity assertion always compares the same model.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api import main
from src.api.backends import OnnxBackend, SklearnBackend, load_backend

pytest.importorskip("onnxruntime")
pytest.importorskip("skl2onnx")

from src.training.onnx_export import convert_pipeline_to_onnx  # noqa: E402

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
JOBLIB_PATH = MODELS_DIR / "model.joblib"

pytestmark = pytest.mark.skipif(
    not JOBLIB_PATH.exists(), reason="Run `python -m src.training.train` first"
)

TEXTS = [
    "Patient presents with acute chest pain radiating to the left arm, elevated "
    "troponin levels, and ECG changes consistent with myocardial infarction.",
    "Colonoscopy revealed a small benign polyp in the sigmoid colon; no dysplasia.",
    "Progressive memory loss, gait disturbance and urinary incontinence over two years.",
    "Bismuth subsalicylate therapy and beta lactamase producing strains were evaluated.",
    "The patient was treated and discharged. The outcome was good.",  # mostly stop words
    "!!!!!!!!!!!!!!!!!!!!",  # no tokens at all
    "Paciente com dor toracica aguda, sudorese e nausea; troponina ELEVADA.",
    "Café, naïve résumé — IL-6 and TNF-α levels were elevated (p<0.05).",
]


@pytest.fixture(scope="module")
def models_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A models/ directory with the current joblib and an ONNX freshly exported from it."""
    import joblib

    directory = tmp_path_factory.mktemp("models")
    shutil.copy(JOBLIB_PATH, directory / "model.joblib")
    onnx_model = convert_pipeline_to_onnx(joblib.load(JOBLIB_PATH))
    (directory / "model.onnx").write_bytes(onnx_model.SerializeToString())
    return directory


@pytest.fixture
def client_for(
    models_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[Callable[..., TestClient]]:
    """Factory: start the real app against ``models_dir`` with a given env value."""
    stack: list[TestClient] = []

    def _make(backend: str | None, directory: Path = models_dir) -> TestClient:
        monkeypatch.setattr(main, "MODELS_DIR", directory)
        if backend is None:
            monkeypatch.delenv("VITORIUM_BACKEND", raising=False)
        else:
            monkeypatch.setenv("VITORIUM_BACKEND", backend)
        client = TestClient(main.app)
        client.__enter__()
        stack.append(client)
        return client

    yield _make
    for client in reversed(stack):
        client.__exit__(None, None, None)


def test_backends_agree_on_labels_and_probabilities(models_dir: Path) -> None:
    sk = SklearnBackend(models_dir / "model.joblib")
    ox = OnnxBackend(models_dir / "model.onnx")

    sk_classes, sk_proba = sk.predict_proba(TEXTS)
    ox_classes, ox_proba = ox.predict_proba(TEXTS)

    assert ox_classes == sk_classes
    assert ox_proba.shape == sk_proba.shape == (len(TEXTS), len(sk_classes))
    np.testing.assert_allclose(ox_proba, sk_proba, rtol=0, atol=1e-5)
    assert [sk_classes[i] for i in ox_proba.argmax(axis=1)] == [
        sk_classes[i] for i in sk_proba.argmax(axis=1)
    ]
    # Batched and single-row calls must match too (the API sends one text per call).
    for i, text in enumerate(TEXTS):
        _, single = ox.predict_proba([text])
        np.testing.assert_allclose(single[0], ox_proba[i], rtol=0, atol=1e-6)


def test_health_reports_onnx_backend_by_default(client_for: Callable[..., TestClient]) -> None:
    client = client_for(None)
    body = client.get("/health").json()
    assert body == {"status": "ok", "model_loaded": True, "backend": "onnx"}
    metrics = client.get("/metrics").text
    assert 'vitorium_model_info{backend="onnx"} 1.0' in metrics


def test_env_var_forces_sklearn(client_for: Callable[..., TestClient]) -> None:
    client = client_for("sklearn")
    assert client.get("/health").json()["backend"] == "sklearn"
    assert 'vitorium_model_info{backend="sklearn"} 1.0' in client.get("/metrics").text


def test_predict_is_identical_across_backends(client_for: Callable[..., TestClient]) -> None:
    payload = {"texto": TEXTS[0]}
    onnx_body = client_for("onnx").post("/predict", json=payload).json()
    sklearn_body = client_for("sklearn").post("/predict", json=payload).json()

    assert (
        set(onnx_body)
        == set(sklearn_body)
        == {
            "urgencia",
            "confianca",
            "probabilidades",
            "latencia_ms",
        }
    )
    assert onnx_body["urgencia"] == sklearn_body["urgencia"]
    for cls, p in sklearn_body["probabilidades"].items():
        assert onnx_body["probabilidades"][cls] == pytest.approx(p, abs=1e-5)


def test_falls_back_to_sklearn_without_onnx_file(
    client_for: Callable[..., TestClient], tmp_path: Path
) -> None:
    shutil.copy(JOBLIB_PATH, tmp_path / "model.joblib")
    client = client_for("onnx", tmp_path)
    assert client.get("/health").json()["backend"] == "sklearn"


def test_invalid_backend_is_rejected(
    client_for: Callable[..., TestClient], models_dir: Path
) -> None:
    with pytest.raises(ValueError, match="VITORIUM_BACKEND"):
        load_backend("tensorflow", models_dir)
    with pytest.raises(ValueError, match="VITORIUM_BACKEND"):
        client_for("tensorflow")  # the API refuses to start


def test_no_artifacts_means_no_backend(tmp_path: Path) -> None:
    assert load_backend(None, tmp_path) is None
    assert load_backend("sklearn", tmp_path) is None
