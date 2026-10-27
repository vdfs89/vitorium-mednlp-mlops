"""Audit of the *stock* ONNX export used up to v0.1.1 (backs the README errata).

Reproduces the original conversion (skl2onnx.convert_sklearn, zipmap=False)
from the versioned models/model.joblib and measures its parity against the
sklearn pipeline on the whole test split. This is the export that produced the
7.70x speedup published until v0.1.1; it is NOT equivalent to the model (it
drops the English stop words), which is why src/training/onnx_export.py exists.

Usage (the stock graph only loads with the en_US.UTF-8 locale installed):
    LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8 PYTHONPATH=. \
        python scripts/audit_stock_onnx_export.py docs/evidence/onnx_stock_export_parity.json
"""

from __future__ import annotations

import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import onnxruntime as ort
import pandas as pd
import skl2onnx
import sklearn
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import StringTensorType

ROOT = Path(__file__).resolve().parents[1]


def main(out_path: Path) -> dict:
    pipe = joblib.load(ROOT / "models" / "model.joblib")
    onx = convert_sklearn(
        pipe,
        initial_types=[("texto", StringTensorType([None, 1]))],
        options={id(pipe): {"zipmap": False}},
    )
    sess = ort.InferenceSession(onx.SerializeToString(), providers=["CPUExecutionProvider"])

    texts = pd.read_csv(ROOT / "data" / "raw" / "medical_tc_test.csv")["medical_abstract"]
    texts = texts.astype(str).tolist()
    sk_proba = pipe.predict_proba(texts)
    _, onnx_proba = sess.run(None, {"texto": np.array(texts, dtype=object).reshape(-1, 1)})
    onnx_proba = np.asarray(onnx_proba, dtype=np.float64)

    row_diff = np.abs(onnx_proba - sk_proba).max(axis=1)
    same_label = pipe.classes_[sk_proba.argmax(1)] == pipe.classes_[onnx_proba.argmax(1)]

    result = {
        "what": (
            "Exportacao ONNX padrao (skl2onnx.convert_sklearn, zipmap=False), "
            "usada ate a v0.1.1 e que gerou o 7,70x publicado entao"
        ),
        "n_texts": len(texts),
        "texts_source": "data/raw/medical_tc_test.csv (todas as linhas)",
        "label_agreement": round(float(same_label.mean()), 4),
        "label_disagreements": int((~same_label).sum()),
        "max_abs_proba_diff": float(row_diff.max()),
        "mean_abs_proba_diff": float(row_diff.mean()),
        "rows_above_1e-5": int((row_diff > 1e-5).sum()),
        "note": (
            "Requer o locale en_US.UTF-8 para carregar (instalado so no ambiente "
            "desta medicao); ele nao existe na imagem python:3.11-slim."
        ),
        "skl2onnx_version": skl2onnx.__version__,
        "onnxruntime_version": ort.__version__,
        "sklearn_version": sklearn.__version__,
        "python": platform.python_version(),
        "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    return result


if __name__ == "__main__":
    print(json.dumps(main(Path(sys.argv[1])), indent=2, ensure_ascii=False))
