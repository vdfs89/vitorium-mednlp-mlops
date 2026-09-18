"""Airflow DAG — VITORIUM training/retraining pipeline.

Three tasks, matching the Fase 3 rubric ("DAG funcional realizando as etapas
de ingestao e treino"):
    ingest_data      -> load and preprocess the raw corpus into data/processed/
    train_model       -> fit the TF-IDF+LogisticRegression pipeline
    save_model_report -> persist metrics.json as the run's artifact summary

This DAG calls the exact same functions used by `python -m src.training.*`
(no duplicated logic) so a local run and an Airflow run stay in sync.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime

from airflow.decorators import dag, task

logger = logging.getLogger(__name__)

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
def vitorium_train_pipeline():
    @task
    def ingest_data() -> None:
        from src.training.preprocess import main as preprocess_main

        preprocess_main()

    @task
    def train_model() -> dict:
        from src.training.train import main as train_main

        train_main()

        with open("models/metrics.json") as f:
            metrics = json.load(f)
        return {"accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"]}

    @task
    def save_model_report(metrics: dict) -> None:
        logger.info(
            "VITORIUM retrain complete — accuracy=%.4f macro_f1=%.4f",
            metrics["accuracy"],
            metrics["macro_f1"],
        )

    ingest = ingest_data()
    metrics = train_model()
    save_model_report(metrics)
    ingest >> metrics


vitorium_train_pipeline()
