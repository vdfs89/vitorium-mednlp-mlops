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
    Gauge,
    Histogram,
    generate_latest,
)
from starlette.routing import Match

REQUEST_COUNT = Counter(
    "vitorium_requests_total",
    "Total de requisicoes recebidas pela API",
    ["method", "endpoint", "status_code"],
)

REQUEST_LATENCY = Histogram(
    "vitorium_request_latency_seconds",
    "Latencia das requisicoes em segundos",
    ["method", "endpoint"],
    # 1 ms e 2,5 ms existem porque a API responde em ~2 ms: sem eles todo o
    # trafego caia no primeiro bucket e o p95 do Grafana virava interpolacao.
    buckets=(0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)

PREDICTION_COUNT = Counter(
    "vitorium_predictions_total",
    "Total de predicoes por classe de urgencia",
    ["urgencia"],
)

# Info-style gauge (value is always 1): lets Grafana/alerts see which inference
# engine a replica is serving, e.g. after a silent ONNX -> sklearn fallback.
MODEL_INFO = Gauge(
    "vitorium_model_info",
    "Backend de inferencia ativo (valor sempre 1)",
    ["backend"],
)


def set_model_info(backend: str) -> None:
    """Expose exactly one active backend label (clears the previous one)."""
    MODEL_INFO.clear()
    MODEL_INFO.labels(backend=backend).set(1)


UNMATCHED_ENDPOINT = "unmatched"


def route_label(app: FastAPI, request: Request) -> str:
    """Return the route *template* for the label, never the raw URL path.

    Using ``request.url.path`` would create one Prometheus series per distinct
    URL (e.g. every 404 probe), an unbounded-cardinality leak. Known routes keep
    their template (``/predict``, ``/health``, ``/metrics``); anything else is
    collapsed into a single ``unmatched`` label.
    """
    for route in app.router.routes:
        match, _ = route.matches(request.scope)
        if match == Match.FULL:
            return getattr(route, "path", UNMATCHED_ENDPOINT)
    return UNMATCHED_ENDPOINT


def install_metrics_middleware(app: FastAPI) -> None:
    @app.middleware("http")
    async def prometheus_middleware(request: Request, call_next: Callable) -> Response:
        """Record count and latency for every request, including unhandled errors.

        An exception raised inside the endpoint escapes ``call_next`` before any
        response exists; Starlette's ServerErrorMiddleware then answers 500. The
        ``finally`` block records that request as status 500 and re-raises, so
        the error is both counted and still handled (and logged) upstream.
        """
        endpoint = route_label(app, request)
        start = time.perf_counter()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            response.headers["X-Process-Time-Ms"] = (
                f"{(time.perf_counter() - start) * 1000:.2f}"
            )
            return response
        finally:
            duration = time.perf_counter() - start
            REQUEST_LATENCY.labels(method=request.method, endpoint=endpoint).observe(duration)
            REQUEST_COUNT.labels(
                method=request.method,
                endpoint=endpoint,
                status_code=status_code,
            ).inc()

    @app.get("/metrics", include_in_schema=False)
    def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
