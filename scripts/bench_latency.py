"""Latency benchmark for the VITORIUM /predict endpoint.

Etapa 1: baseline against the running (non-containerized or containerized)
FastAPI server. Etapa 4 reruns this same script against the ONNX-optimized
model to produce a genuine before/after comparison -- no numbers here are
estimated, they are measured against a live server.

Usage:
    uvicorn src.api.main:app --port 8000 &
    python scripts/bench_latency.py --url http://127.0.0.1:8000 --n 200
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import httpx

SAMPLE_LAUDO = (
    "Patient presents with acute chest pain radiating to the left arm, "
    "elevated troponin levels, and ECG changes consistent with myocardial "
    "infarction. Immediate cardiology consultation requested."
)


def percentile(values: list[float], pct: float) -> float:
    values = sorted(values)
    idx = min(int(len(values) * pct), len(values) - 1)
    return values[idx]


def run(url: str, n: int, warmup: int) -> dict:
    with httpx.Client(base_url=url, timeout=10.0) as client:
        for _ in range(warmup):
            client.post("/predict", json={"texto": SAMPLE_LAUDO})

        latencies_ms = []
        for _ in range(n):
            start = time.perf_counter()
            resp = client.post("/predict", json={"texto": SAMPLE_LAUDO})
            elapsed_ms = (time.perf_counter() - start) * 1000
            resp.raise_for_status()
            latencies_ms.append(elapsed_ms)

    return {
        "n": n,
        "p50_ms": round(statistics.median(latencies_ms), 3),
        "p95_ms": round(percentile(latencies_ms, 0.95), 3),
        "p99_ms": round(percentile(latencies_ms, 0.99), 3),
        "mean_ms": round(statistics.mean(latencies_ms), 3),
        "min_ms": round(min(latencies_ms), 3),
        "max_ms": round(max(latencies_ms), 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--out", default=None, help="Optional JSON output path")
    args = parser.parse_args()

    result = run(args.url, args.n, args.warmup)
    print(json.dumps(result, indent=2))

    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
