"""Etapa 4 — export the trained sklearn pipeline to ONNX for faster inference.

Usage:
    python scripts/export_onnx.py

Writes models/model.onnx. Run scripts/bench_latency.py before and after
swapping the API's inference backend to produce the required latency
comparison (original vs. optimized) for the README/video.
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import StringTensorType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


def main() -> None:
    pipeline = joblib.load(MODELS_DIR / "model.joblib")

    initial_type = [("texto", StringTensorType([None, 1]))]
    onnx_model = convert_sklearn(
        pipeline,
        initial_types=initial_type,
        options={id(pipeline): {"zipmap": False}},
    )

    out_path = MODELS_DIR / "model.onnx"
    with open(out_path, "wb") as f:
        f.write(onnx_model.SerializeToString())

    logger.info("Wrote ONNX model to %s (%d bytes)", out_path, out_path.stat().st_size)


if __name__ == "__main__":
    main()
