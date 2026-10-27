"""VITORIUM — inference API for medical text triage.

Routes:
    POST /predict  - classify a laudo/report text into normal/atencao/urgente
    GET  /health   - liveness/readiness probe (also reports the active backend)
    GET  /metrics  - Prometheus scrape endpoint (installed via metrics.py)

The inference engine is chosen at startup by ``VITORIUM_BACKEND``
(``onnx`` | ``sklearn``; unset = onnx with sklearn fallback). See backends.py.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Response, status

from .backends import BACKEND_ENV_VAR, InferenceBackend, load_backend
from .metrics import PREDICTION_COUNT, install_metrics_middleware, set_model_info
from .schemas import HealthOutput, LaudoInput, PredictionOutput

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"

_backend: InferenceBackend | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global _backend
    requested = os.environ.get(BACKEND_ENV_VAR)
    try:
        _backend = load_backend(requested, MODELS_DIR)
    except ValueError:
        logger.exception("Refusing to start: invalid inference backend configuration.")
        raise
    set_model_info(_backend.name if _backend is not None else "none")
    yield
    _backend = None


app = FastAPI(
    title="VITORIUM",
    description="End-to-End MLOps Pipeline for Medical Text Classification",
    version="0.1.0",
    lifespan=lifespan,
)
install_metrics_middleware(app)


@app.get("/health", response_model=HealthOutput)
def health(response: Response) -> HealthOutput:
    """Readiness probe: 200 only when a backend is loaded and /predict can answer.

    Returning 503 without a model makes the Docker HEALTHCHECK (and any load
    balancer) mark the container unhealthy instead of routing traffic to an
    instance whose /predict would fail with 503.
    """
    if _backend is None:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return HealthOutput(status="unavailable", model_loaded=False, backend="none")
    return HealthOutput(status="ok", model_loaded=True, backend=_backend.name)


@app.post("/predict", response_model=PredictionOutput)
def predict(payload: LaudoInput) -> PredictionOutput:
    backend = _backend
    if backend is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start = time.perf_counter()
    classes, probas = backend.predict_proba([payload.texto])
    latency_ms = (time.perf_counter() - start) * 1000

    proba = probas[0]
    idx = int(proba.argmax())
    urgencia = classes[idx]

    PREDICTION_COUNT.labels(urgencia=urgencia).inc()

    return PredictionOutput(
        urgencia=urgencia,
        confianca=float(proba[idx]),
        probabilidades={cls: float(p) for cls, p in zip(classes, proba, strict=True)},
        latencia_ms=latency_ms,
    )
