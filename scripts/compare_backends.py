"""Etapa 4 — end-to-end (HTTP) latency: API serving sklearn vs. API serving ONNX.

Why this exists: ``onnx_latency_comparison.json`` measures pure in-process
inference. What a client feels is the full request: HTTP, FastAPI routing,
Pydantic validation, Prometheus middleware *and* inference. This script starts
two real uvicorn servers from the same code, differing only in
``VITORIUM_BACKEND``, and benchmarks ``POST /predict`` on both with the exact
same logic as ``scripts/bench_latency.py``.

Drift control: the backends are measured in several interleaved rounds
(round r runs sklearn->onnx when r is even and onnx->sklearn when odd), and the
summary uses the median across rounds. Every per-round raw value is saved.

Usage:
    PYTHONPATH=. python scripts/compare_backends.py              # 5 rounds x n=300
    PYTHONPATH=. python scripts/compare_backends.py --rounds 3 --n 100

Writes models/api_latency_comparison.json.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from contextlib import ExitStack
from datetime import UTC, datetime
from pathlib import Path
from typing import IO

import httpx
import onnxruntime
import sklearn
from bench_latency import SAMPLE_LAUDO, run  # same directory as this script

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("compare_backends")
# httpx logs every request at INFO; that would run inside the timed window and
# differ from a standalone bench_latency.py run, so keep it quiet.
logging.getLogger("httpx").setLevel(logging.WARNING)

ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "models" / "api_latency_comparison.json"
BACKENDS = ("sklearn", "onnx")
METRICS = ("p50_ms", "p95_ms", "p99_ms", "mean_ms")


def start_server(backend: str, port: int, log_file: IO[str]) -> subprocess.Popen:
    """Launch uvicorn like the Dockerfile CMD (host/port aside), pinned to one backend."""
    env = {**os.environ, "VITORIUM_BACKEND": backend, "PYTHONPATH": str(ROOT)}
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "src.api.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        cwd=ROOT,
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )


def wait_healthy(url: str, expected_backend: str, timeout_s: float = 60.0) -> dict:
    """Block until /health answers and confirm the server runs the requested backend.

    The backend check matters: the API silently falls back to sklearn when ONNX
    cannot load, which would turn this benchmark into sklearn-vs-sklearn.
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            health = httpx.get(f"{url}/health", timeout=2.0).json()
        except httpx.HTTPError:
            time.sleep(0.25)
            continue
        if not health.get("model_loaded") or health.get("backend") != expected_backend:
            raise RuntimeError(f"{url} is serving {health}, expected backend={expected_backend}")
        return health
    raise TimeoutError(f"{url} did not become healthy within {timeout_s}s")


def cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--base-port", type=int, default=8101)
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    args = parser.parse_args()

    urls = {b: f"http://127.0.0.1:{args.base_port + i}" for i, b in enumerate(BACKENDS)}
    per_round: dict[str, list[dict]] = {b: [] for b in BACKENDS}

    with ExitStack() as stack:
        log_dir = Path(stack.enter_context(tempfile.TemporaryDirectory()))
        for i, backend in enumerate(BACKENDS):
            log_file = stack.enter_context(open(log_dir / f"uvicorn_{backend}.log", "w"))
            proc = start_server(backend, args.base_port + i, log_file)
            stack.callback(proc.wait, 10)
            stack.callback(proc.terminate)

        try:
            for backend in BACKENDS:
                logger.info("health %s: %s", backend, wait_healthy(urls[backend], backend))
        except Exception:
            for backend in BACKENDS:
                logger.error(
                    "--- uvicorn %s log ---\n%s",
                    backend,
                    (log_dir / f"uvicorn_{backend}.log").read_text(),
                )
            raise

        # Same input on both servers must give the same answer (within float32).
        samples = {
            b: httpx.post(f"{urls[b]}/predict", json={"texto": SAMPLE_LAUDO}).json()
            for b in BACKENDS
        }
        max_diff = max(
            abs(samples["sklearn"]["probabilidades"][c] - samples["onnx"]["probabilidades"][c])
            for c in samples["sklearn"]["probabilidades"]
        )
        if samples["sklearn"]["urgencia"] != samples["onnx"]["urgencia"] or max_diff > 1e-5:
            raise RuntimeError(f"Backends disagree on the benchmark text: {samples}")

        for r in range(args.rounds):
            order = BACKENDS if r % 2 == 0 else tuple(reversed(BACKENDS))
            for backend in order:
                result = run(urls[backend], args.n, args.warmup)
                per_round[backend].append(result)
                logger.info("round %d %-7s %s", r + 1, backend, result)

    summary = {
        b: {m: round(statistics.median(x[m] for x in per_round[b]), 3) for m in METRICS}
        for b in BACKENDS
    }
    speedup = {
        f"{m.removesuffix('_ms')}_x": round(summary["sklearn"][m] / summary["onnx"][m], 3)
        for m in METRICS
    }
    result = {
        "summary_median_across_rounds": summary,
        "speedup_sklearn_over_onnx": speedup,
        "delta_p50_ms": round(summary["sklearn"]["p50_ms"] - summary["onnx"]["p50_ms"], 3),
        "per_round": per_round,
        "n": args.n,
        "rounds": args.rounds,
        "warmup": args.warmup,
        "endpoint": "POST /predict",
        "payload_chars": len(SAMPLE_LAUDO),
        "sample_prediction": {
            b: {k: samples[b][k] for k in ("urgencia", "probabilidades")} for b in BACKENDS
        },
        "sample_max_abs_proba_diff": max_diff,
        "method": (
            "Two uvicorn servers (same code, same machine, 1 worker each, default "
            "logging as in the Dockerfile CMD) differing only in VITORIUM_BACKEND; "
            "/health verified to report the requested backend. Each round runs "
            "scripts/bench_latency.run (sequential keep-alive httpx client, warmup "
            "requests discarded, client-side wall time per request) on both, "
            "alternating order per round; summary = median across rounds."
        ),
        "python": platform.python_version(),
        "sklearn_version": sklearn.__version__,
        "onnxruntime_version": onnxruntime.__version__,
        "httpx_version": httpx.__version__,
        "cpu_model": cpu_model(),
        "cpu_count": os.cpu_count(),
        "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    logger.info("summary %s | speedup %s", json.dumps(summary), json.dumps(speedup))
    logger.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
