"""Train the VITORIUM triage baseline: TF-IDF + Logistic Regression.

Deliberately a "leve" (lightweight) model per the Tech Challenge Fase 3
requirements — the graded surface here is MLOps plumbing (CI/CD, Airflow,
monitoring, latency optimization), not model sophistication.

Usage:
    python -m src.training.train

Reads data/processed/triage_{train,test}.csv (see preprocess.py),
fits the pipeline, evaluates on the held-out test split, and writes:
    models/model.joblib
    models/metrics.json
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
MODELS_DIR = Path(__file__).resolve().parents[2] / "models"

RANDOM_SEED = 42


def build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=20_000,
                    ngram_range=(1, 2),
                    min_df=2,
                    stop_words="english",
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    random_state=RANDOM_SEED,
                    class_weight="balanced",
                ),
            ),
        ]
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(PROCESSED_DIR / "triage_train.csv")
    test_df = pd.read_csv(PROCESSED_DIR / "triage_test.csv")

    pipeline = build_pipeline()

    t0 = time.perf_counter()
    pipeline.fit(train_df["texto"], train_df["urgencia"])
    train_seconds = time.perf_counter() - t0
    logger.info("Trained in %.2fs on %d rows", train_seconds, len(train_df))

    preds = pipeline.predict(test_df["texto"])
    accuracy = accuracy_score(test_df["urgencia"], preds)
    macro_f1 = f1_score(test_df["urgencia"], preds, average="macro")
    report = classification_report(test_df["urgencia"], preds, output_dict=True)

    logger.info("Test accuracy: %.4f | macro F1: %.4f", accuracy, macro_f1)

    joblib.dump(pipeline, MODELS_DIR / "model.joblib")

    metrics = {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "train_seconds": train_seconds,
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "classification_report": report,
        "random_seed": RANDOM_SEED,
    }
    with open(MODELS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(
        "Saved model to %s and metrics to %s",
        MODELS_DIR / "model.joblib",
        MODELS_DIR / "metrics.json",
    )


if __name__ == "__main__":
    main()
