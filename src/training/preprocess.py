"""Preprocessing for the VITORIUM triage classifier.

Maps the 5 original condition-type labels of the Medical Abstracts TC Corpus
(sebischair/Medical-Abstracts-TC-Corpus) onto a 3-tier urgency proxy used as
the illustrative business framing for this Tech Challenge.

IMPORTANT — honesty note (see README "Limitações"):
The urgency labels (normal / atencao / urgente) are a heuristic regrouping of
the original condition-type classes, created for MLOps teaching purposes.
They do NOT reflect real clinical triage (e.g. Manchester, ESI) and must not
be used for medical decision-making.

Original classes (1-5), per the corpus README:
    1 = neoplasms
    2 = digestive system diseases
    3 = nervous system diseases
    4 = cardiovascular diseases
    5 = general pathological conditions

Mapping applied here:
    urgente -> cardiovascular (4) + nervous system (3)
               [acute/emergency-pattern conditions]
    atencao -> neoplasms (1) + digestive system (2)
               [needs follow-up, rarely an immediate emergency]
    normal  -> general pathological (5)
               [generic / non-specific catch-all]
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"

LABEL_NAMES = {
    1: "neoplasms",
    2: "digestive_system_diseases",
    3: "nervous_system_diseases",
    4: "cardiovascular_diseases",
    5: "general_pathological_conditions",
}

# condition_label (int) -> urgency tier
URGENCY_MAP = {
    4: "urgente",
    3: "urgente",
    1: "atencao",
    2: "atencao",
    5: "normal",
}

URGENCY_TIERS = ["normal", "atencao", "urgente"]


def load_raw(split: str) -> pd.DataFrame:
    """Load the raw corpus split ('train' or 'test')."""
    path = RAW_DIR / f"medical_tc_{split}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download it from "
            "https://github.com/sebischair/Medical-Abstracts-TC-Corpus first."
        )
    df = pd.read_csv(path)
    logger.info("Loaded %d rows from %s", len(df), path)
    return df


def map_urgency(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the heuristic condition->urgency mapping and clean text."""
    out = df.copy()
    out["condition_name"] = out["condition_label"].map(LABEL_NAMES)
    out["urgencia"] = out["condition_label"].map(URGENCY_MAP)
    out["texto"] = out["medical_abstract"].astype(str).str.strip()

    missing = out["urgencia"].isna().sum()
    if missing:
        raise ValueError(f"{missing} rows failed to map to an urgency tier")

    return out[["texto", "urgencia", "condition_name", "condition_label"]]


def build_processed_splits() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = map_urgency(load_raw("train"))
    test = map_urgency(load_raw("test"))
    return train, test


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    train, test = build_processed_splits()

    train.to_csv(PROCESSED_DIR / "triage_train.csv", index=False)
    test.to_csv(PROCESSED_DIR / "triage_test.csv", index=False)

    for name, split in [("train", train), ("test", test)]:
        dist = split["urgencia"].value_counts(normalize=True).round(3).to_dict()
        logger.info("%s urgency distribution: %s", name, dist)

    logger.info(
        "Wrote %d train rows and %d test rows to %s",
        len(train),
        len(test),
        PROCESSED_DIR,
    )


if __name__ == "__main__":
    main()
