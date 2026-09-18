"""Prometheus instrumentation for the VITORIUM inference API.

Exposes, at minimum, the metrics required by the Fase 3 rubric:
total request count, request latency, and error rate (derived from status
labels on REQUEST_COUNT).
"""

from __future__ import annotations

import time
from collections.abc import Callable

from fastapi import FastAPI, Request, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Histogram,
    generate_latest,
)

REQUEST_COUNT = Counter(
    "vitorium_requests_total",
    "Total de requisicoes recebidas pela API",
    ["method", "endpoint", "status_code"],
)

REQUEST_LATENCY = Histogram(
    "vitorium_request_latency_seconds",
    "Latencia das requisicoes em segundos",
    ["method", "endpoint"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)

PREDICTION_COUNT = Counter(
    "vitorium_predictions_total",
    "Total de predicoes por classe de urgencia",
    ["urgencia"],
)


def install_metrics_middleware(app: FastAPI) -> None:
    @app.middleware("http")
    async def prometheus_middleware(request: Request, call_next: Callable) -> Response:
        start = time.perf_counter()
        response = await call_next(request)
        duration = time.perf_counter() - start

        endpoint = request.url.path
        REQUEST_LATENCY.labels(method=request.method, endpoint=endpoint).observe(duration)
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=endpoint,
            status_code=response.status_code,
        ).inc()
        response.headers["X-Process-Time-Ms"] = f"{duration * 1000:.2f}"
        return response

    @app.get("/metrics", include_in_schema=False)
    def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
