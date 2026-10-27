"""Etapa 4 — export the trained sklearn pipeline to ONNX and measure the gain.

Usage:
    PYTHONPATH=. python scripts/export_onnx.py              # export + parity + bench
    PYTHONPATH=. python scripts/export_onnx.py --skip-bench # export + parity (CI)

Writes:
    models/model.onnx                      (always)
    models/onnx_latency_comparison.json    (unless --skip-bench)

Parity is checked on every run (the script exits non-zero if the ONNX model
disagrees with sklearn), because a faster model that answers differently is
not an optimization. The latency numbers are pure in-process inference (no
HTTP); scripts/compare_backends.py measures the end-to-end API instead.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import onnxruntime
import pandas as pd
import sklearn

from src.api.backends import OnnxBackend, SklearnBackend
from src.training.onnx_export import convert_pipeline_to_onnx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
RAW_TEST_CSV = ROOT / "data" / "raw" / "medical_tc_test.csv"

PROBA_TOLERANCE = 1e-5
FALLBACK_TEXTS = [
    "Patient presents with acute chest pain radiating to the left arm, elevated "
    "troponin levels, and ECG changes consistent with myocardial infarction.",
    "Routine follow-up of a benign colonic polyp; no dysplasia was identified.",
    "Progressive memory loss and gait disturbance over two years.",
]


def load_eval_texts() -> tuple[list[str], str]:
    """Real abstracts from the held-out raw test split when available."""
    if RAW_TEST_CSV.exists():
        texts = pd.read_csv(RAW_TEST_CSV)["medical_abstract"].astype(str).str.strip().tolist()
        return texts, "data/raw/medical_tc_test.csv (all rows)"
    return FALLBACK_TEXTS, "built-in sample texts (raw test split not found)"


def check_parity(sk: SklearnBackend, ox: OnnxBackend, texts: list[str]) -> dict:
    sk_classes, sk_p = sk.predict_proba(texts)
    ox_classes, ox_p = ox.predict_proba(texts)
    if sk_classes != ox_classes:
        raise SystemExit(f"Class order mismatch: sklearn={sk_classes} onnx={ox_classes}")
    diff = np.abs(sk_p - ox_p).max(axis=1)
    label_agreement = float((sk_p.argmax(axis=1) == ox_p.argmax(axis=1)).mean())
    return {
        "n_texts": len(texts),
        "max_abs_proba_diff": float(diff.max()),
        "rows_above_tolerance": int((diff > PROBA_TOLERANCE).sum()),
        "tolerance": PROBA_TOLERANCE,
        "label_agreement": label_agreement,
    }


def bench(sk: SklearnBackend, ox: OnnxBackend, texts: list[str], n: int, warmup: int) -> dict:
    """Single-text predict_proba calls (the API's call pattern), interleaved per text."""
    sample = [texts[i % len(texts)] for i in range(n)]
    for text in sample[:warmup]:
        sk.predict_proba([text])
        ox.predict_proba([text])

    sk_ms: list[float] = []
    ox_ms: list[float] = []
    for text in sample:
        t0 = time.perf_counter()
        sk.predict_proba([text])
        sk_ms.append((time.perf_counter() - t0) * 1000)
        t0 = time.perf_counter()
        ox.predict_proba([text])
        ox_ms.append((time.perf_counter() - t0) * 1000)

    def q(values: list[float], pct: float) -> float:
        ordered = sorted(values)
        return ordered[min(int(len(ordered) * pct), len(ordered) - 1)]

    return {
        "sklearn_mean_ms": round(statistics.mean(sk_ms), 4),
        "onnx_mean_ms": round(statistics.mean(ox_ms), 4),
        "speedup_x": round(statistics.mean(sk_ms) / statistics.mean(ox_ms), 2),
        "sklearn_p50_ms": round(statistics.median(sk_ms), 4),
        "onnx_p50_ms": round(statistics.median(ox_ms), 4),
        "speedup_p50_x": round(statistics.median(sk_ms) / statistics.median(ox_ms), 2),
        "sklearn_p95_ms": round(q(sk_ms, 0.95), 4),
        "onnx_p95_ms": round(q(ox_ms, 0.95), 4),
    }


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
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--skip-bench", action="store_true")
    args = parser.parse_args()

    joblib_path = MODELS_DIR / "model.joblib"
    onnx_path = MODELS_DIR / "model.onnx"

    # The candidate is written next to the target and only promoted (atomic
    # os.replace) after the parity check passes: a failed export leaves the
    # last approved models/model.onnx byte-for-byte untouched.
    candidate_path = onnx_path.with_name(onnx_path.name + ".candidate")
    onnx_model = convert_pipeline_to_onnx(joblib.load(joblib_path))
    candidate_path.write_bytes(onnx_model.SerializeToString())

    sk, ox = SklearnBackend(joblib_path), OnnxBackend(candidate_path)
    texts, source = load_eval_texts()
    parity = check_parity(sk, ox, texts)
    logger.info("Parity on %s: %s", source, parity)
    if parity["rows_above_tolerance"] or parity["label_agreement"] < 1.0:
        candidate_path.unlink(missing_ok=True)
        sys.exit("ONNX candidate is not equivalent to the sklearn pipeline; "
                 f"kept the previous {onnx_path.name} unchanged.")

    os.replace(candidate_path, onnx_path)
    logger.info("Promoted ONNX model to %s (%d bytes)", onnx_path, onnx_path.stat().st_size)
    ox = OnnxBackend(onnx_path)

    if args.skip_bench:
        return

    result = {
        "n": args.n,
        **bench(sk, ox, texts, args.n, args.warmup),
        "labels_match": parity["label_agreement"] == 1.0,
        "warmup": args.warmup,
        "parity": {**parity, "texts_source": source},
        "method": (
            "In-process predict_proba on one text per call (same call pattern as "
            "/predict), sklearn and ONNX interleaved per text after warmup; texts "
            "cycled from the evaluation set. No HTTP; see api_latency_comparison.json."
        ),
        "python": platform.python_version(),
        "sklearn_version": sklearn.__version__,
        "onnxruntime_version": onnxruntime.__version__,
        "cpu_model": cpu_model(),
        "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    out = MODELS_DIR / "onnx_latency_comparison.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    logger.info("Wrote %s: %s", out, json.dumps(result))


if __name__ == "__main__":
    main()
