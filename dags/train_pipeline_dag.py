"""Airflow DAG — VITORIUM training/retraining pipeline.

Four tasks, matching the rubric ("DAG funcional realizando as etapas de
ingestao e treino") plus the serving artifact the API actually uses:
    ingest_data       -> load and preprocess the raw corpus into data/processed/
    train_model       -> fit the TF-IDF+LogisticRegression pipeline
    export_onnx       -> re-export models/model.onnx from the new model.joblib and
                         fail the run if the ONNX graph is not equivalent to sklearn
    save_model_report -> log the run's metrics (models/metrics.json) as its summary

Without export_onnx a retrain would update model.joblib while the API (which
serves ONNX by default) kept answering with the previous model.onnx.

This DAG calls the exact same functions used by `python -m src.training.*`
(no duplicated logic) so a local run and an Airflow run stay in sync.

Execution evidence: docs/evidence/airflow_dags_test.log, reproducible with
scripts/run_airflow_dag_test.sh (`airflow dags test`).
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from airflow.decorators import dag, task

logger = logging.getLogger(__name__)

# The DAG lives in <project_root>/dags/, so the project root is one level up.
# Every path is derived from this file, never from the CWD: Airflow workers do
# not guarantee the working directory a task runs in.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRICS_PATH = PROJECT_ROOT / "models" / "metrics.json"

# Airflow only puts the dags_folder on sys.path, not the project root, so
# `import src...` inside the tasks would fail unless PYTHONPATH is set by the
# deployment. The preferred setup is PYTHONPATH=<project_root> (done by
# scripts/run_airflow_dag_test.sh); this idempotent fallback keeps the DAG
# runnable when the repo's dags/ folder is mounted into Airflow as-is.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

default_args = {
    "owner": "vitorium",
    "retries": 1,
}


@dag(
    dag_id="vitorium_train_pipeline",
    description="Ingestao -> treino -> salvamento do classificador de triagem VITORIUM",
    schedule="@weekly",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    default_args=default_args,
    tags=["vitorium", "mlops", "fase3"],
)
def vitorium_train_pipeline() -> None:
    @task
    def ingest_data() -> None:
        """Rebuild data/processed/ from data/raw/ (heuristic urgency mapping)."""
        from src.training.preprocess import main as preprocess_main

        preprocess_main()

    @task
    def train_model() -> dict[str, float]:
        """Fit and evaluate the model; return only the headline metrics via XCom.

        The full metrics.json (classification report included) stays on disk;
        XCom carries just two floats to keep the metadata DB small.
        """
        from src.training.train import main as train_main

        train_main()

        with METRICS_PATH.open() as f:
            metrics = json.load(f)
        return {"accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"]}

    @task
    def export_onnx() -> None:
        """Re-export model.onnx from the freshly trained model.joblib.

        Runs scripts/export_onnx.py as a subprocess (it owns its CLI parsing) with
        --skip-bench: it writes models/model.onnx and checks parity against the
        sklearn pipeline on the whole test split, exiting non-zero on any
        mismatch -- which fails this task and keeps a non-equivalent artifact
        from being reported as a successful retrain.
        """
        subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / "export_onnx.py"), "--skip-bench"],
            check=True,
            cwd=PROJECT_ROOT,
        )

    @task
    def save_model_report(metrics: dict[str, float]) -> None:
        """Log the headline metrics so they appear in the Airflow task log."""
        logger.info(
            "VITORIUM retrain complete — accuracy=%.4f macro_f1=%.4f",
            metrics["accuracy"],
            metrics["macro_f1"],
        )

    ingest = ingest_data()
    metrics = train_model()
    onnx_export = export_onnx()
    report = save_model_report(metrics)
    ingest >> metrics >> onnx_export >> report


vitorium_train_pipeline()
