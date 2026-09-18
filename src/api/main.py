"""VITORIUM — inference API for medical text triage.

Routes:
    POST /predict  - classify a laudo/report text into normal/atencao/urgente
    GET  /health   - liveness/readiness probe
    GET  /metrics  - Prometheus scrape endpoint (installed via metrics.py)
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException

from .metrics import PREDICTION_COUNT, install_metrics_middleware
from .schemas import HealthOutput, LaudoInput, PredictionOutput

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "model.joblib"

_pipeline = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global _pipeline
    if MODEL_PATH.exists():
        _pipeline = joblib.load(MODEL_PATH)
        logger.info("Model loaded from %s", MODEL_PATH)
    else:
        logger.warning(
            "Model artifact not found at %s. /predict will return 503.", MODEL_PATH
        )
    yield


app = FastAPI(
    title="VITORIUM",
    description="End-to-End MLOps Pipeline for Medical Text Classification",
    version="0.1.0",
    lifespan=lifespan,
)
install_metrics_middleware(app)


@app.get("/health", response_model=HealthOutput)
def health() -> HealthOutput:
    return HealthOutput(status="ok", model_loaded=_pipeline is not None)


@app.post("/predict", response_model=PredictionOutput)
def predict(payload: LaudoInput) -> PredictionOutput:
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start = time.perf_counter()
    proba = _pipeline.predict_proba([payload.texto])[0]
    latency_ms = (time.perf_counter() - start) * 1000

    classes = _pipeline.classes_
    idx = proba.argmax()
    urgencia = classes[idx]

    PREDICTION_COUNT.labels(urgencia=urgencia).inc()

    return PredictionOutput(
        urgencia=urgencia,
        confianca=float(proba[idx]),
        probabilidades={cls: float(p) for cls, p in zip(classes, proba, strict=True)},
        latencia_ms=latency_ms,
    )
