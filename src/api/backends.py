"""Inference backends for the VITORIUM API (scikit-learn vs. ONNX Runtime).

Why an abstraction: Etapa 4 asks for an "original vs. optimized" latency
comparison. Measuring it honestly end-to-end requires the *same* API code to
serve either artifact, so the only variable between two benchmark runs is the
inference engine. Each backend exposes ``predict_proba(texts)`` returning the
class names and an ``[n_texts, n_classes]`` probability matrix in the same
order, so ``/predict`` does not care which one is active.

Selection is driven by the ``VITORIUM_BACKEND`` env var (see ``load_backend``).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import joblib
import numpy as np

logger = logging.getLogger(__name__)

BACKEND_ENV_VAR = "VITORIUM_BACKEND"
VALID_BACKENDS = ("sklearn", "onnx")
JOBLIB_FILENAME = "model.joblib"
ONNX_FILENAME = "model.onnx"
# Must match src/training/onnx_export.py (kept as a literal so the runtime
# image does not need skl2onnx, which that module imports).
CLASSES_METADATA_KEY = "vitorium_classes"

# Used once at load time to self-check the ONNX artifact (see OnnxBackend).
_PROBE_TEXT = "Patient presents with acute chest pain and elevated troponin levels."


class InferenceBackend(Protocol):
    """What /predict needs from a model, independent of the runtime engine."""

    name: str

    def predict_proba(self, texts: Sequence[str]) -> tuple[list[str], np.ndarray]:
        """Return ``(classes, probas)`` with ``probas.shape == (len(texts), len(classes))``."""
        ...


class SklearnBackend:
    """The original artifact: the fitted scikit-learn Pipeline (``model.joblib``)."""

    name = "sklearn"

    def __init__(self, model_path: Path) -> None:
        self._pipeline = joblib.load(model_path)
        self.classes: list[str] = [str(c) for c in self._pipeline.classes_]

    def predict_proba(self, texts: Sequence[str]) -> tuple[list[str], np.ndarray]:
        return self.classes, np.asarray(self._pipeline.predict_proba(list(texts)))


class OnnxBackend:
    """The optimized artifact: ``model.onnx`` executed by ONNX Runtime (CPU).

    Class names come from the model metadata written at export time; if they
    are absent (an ONNX file produced by another tool), they are read from the
    classifier node in the graph, which requires the ``onnx`` package.
    """

    name = "onnx"

    def __init__(self, model_path: Path) -> None:
        import onnxruntime as ort  # local import: optional at runtime

        self._session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self._input_name = self._session.get_inputs()[0].name
        outputs = self._session.get_outputs()
        # Pick outputs by type rather than by name: probabilities are the float
        # tensor, the predicted label is the string tensor.
        self._proba_name = next(o.name for o in outputs if o.type == "tensor(float)")
        self._label_name = next(o.name for o in outputs if o.type == "tensor(string)")
        self.classes = self._read_classes(model_path)
        self._self_check()

    def _read_classes(self, model_path: Path) -> list[str]:
        meta = self._session.get_modelmeta().custom_metadata_map
        if CLASSES_METADATA_KEY in meta:
            return [str(c) for c in json.loads(meta[CLASSES_METADATA_KEY])]

        import onnx  # only needed for ONNX files without our metadata

        model = onnx.load(str(model_path))
        for node in model.graph.node:
            for attr in node.attribute:
                if attr.name == "classlabels_strings":
                    return [s.decode() for s in onnx.helper.get_attribute_value(attr)]
        raise ValueError(f"Could not determine class labels for {model_path}")

    def _self_check(self) -> None:
        """Fail at startup, not on the first request, if labels and columns disagree.

        The graph's own ``label`` output must equal ``classes[argmax(probas)]``;
        a mismatch means the class list is in a different order than the
        probability columns, which would silently mislabel every prediction.
        """
        probas, labels = self._run([_PROBE_TEXT])
        if probas.shape[1] != len(self.classes):
            raise ValueError(
                f"ONNX model has {probas.shape[1]} probability columns but "
                f"{len(self.classes)} class labels"
            )
        if str(labels[0]) != self.classes[int(probas[0].argmax())]:
            raise ValueError("ONNX class labels are not aligned with probability columns")

    def _run(self, texts: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
        feed = {self._input_name: np.asarray(list(texts), dtype=object).reshape(-1, 1)}
        probas, labels = self._session.run([self._proba_name, self._label_name], feed)
        return np.asarray(probas), np.asarray(labels)

    def predict_proba(self, texts: Sequence[str]) -> tuple[list[str], np.ndarray]:
        probas, _ = self._run(texts)
        return self.classes, probas


def _onnx_runtime_available() -> bool:
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        return False
    return True


def _load_sklearn(models_dir: Path) -> SklearnBackend | None:
    path = models_dir / JOBLIB_FILENAME
    if not path.exists():
        logger.warning("Model artifact not found at %s. /predict will return 503.", path)
        return None
    backend = SklearnBackend(path)
    logger.info("Inference backend: sklearn (%s)", path)
    return backend


def load_backend(requested: str | None, models_dir: Path) -> InferenceBackend | None:
    """Build the inference backend for the API.

    ``requested`` is the raw ``VITORIUM_BACKEND`` value:

    * unset/empty -> ``onnx`` if ``model.onnx`` exists, onnxruntime imports and
      the session loads; otherwise ``sklearn`` (logged as a warning).
    * ``onnx``    -> same as above. Falling back instead of crashing keeps the
      service up; ``/health`` and ``vitorium_model_info`` expose which backend
      actually serves.
    * ``sklearn`` -> always the original pipeline (used as the benchmark baseline).
    * anything else -> ``ValueError`` (the API refuses to start).

    Returns ``None`` when no usable artifact exists (``/predict`` answers 503).
    """
    choice = (requested or "").strip().lower()
    if choice and choice not in VALID_BACKENDS:
        raise ValueError(
            f"Invalid {BACKEND_ENV_VAR}={requested!r}; expected one of {VALID_BACKENDS} "
            "or unset (auto: onnx with sklearn fallback)."
        )

    if choice == "sklearn":
        return _load_sklearn(models_dir)

    onnx_path = models_dir / ONNX_FILENAME
    reason: str
    if not onnx_path.exists():
        reason = f"{onnx_path} not found"
    elif not _onnx_runtime_available():
        reason = "onnxruntime is not installed"
    else:
        try:
            backend = OnnxBackend(onnx_path)
        except Exception as exc:  # any load failure (e.g. missing locale) -> fall back
            reason = f"failed to load {onnx_path}: {exc}"
        else:
            logger.info("Inference backend: onnx (%s)", onnx_path)
            return backend

    logger.warning("ONNX backend unavailable (%s); falling back to sklearn.", reason)
    return _load_sklearn(models_dir)
