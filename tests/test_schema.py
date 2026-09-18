"""Pandera schema validation for the processed dataset.

Guards against silent drift in the data-preparation pipeline: if a future
change to preprocess.py produces nulls, unexpected labels, or empty texts,
this test catches it before the Airflow DAG or the API ever sees it.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from pandera.pandas import Check, Column, DataFrameSchema

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"

triage_schema = DataFrameSchema(
    {
        "texto": Column(str, Check.str_length(min_value=1), nullable=False),
        "urgencia": Column(
            str, Check.isin(["normal", "atencao", "urgente"]), nullable=False
        ),
        "condition_name": Column(str, nullable=False),
        "condition_label": Column(int, Check.isin([1, 2, 3, 4, 5]), nullable=False),
    }
)


@pytest.mark.skipif(
    not (PROCESSED_DIR / "triage_train.csv").exists(),
    reason="Run `python -m src.training.preprocess` first",
)
def test_processed_train_schema() -> None:
    df = pd.read_csv(PROCESSED_DIR / "triage_train.csv")
    triage_schema.validate(df)


@pytest.mark.skipif(
    not (PROCESSED_DIR / "triage_test.csv").exists(),
    reason="Run `python -m src.training.preprocess` first",
)
def test_processed_test_schema() -> None:
    df = pd.read_csv(PROCESSED_DIR / "triage_test.csv")
    triage_schema.validate(df)
