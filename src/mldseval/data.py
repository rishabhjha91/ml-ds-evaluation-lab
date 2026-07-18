"""Synthetic dataset generation and data validation.

The dataset is a reproducible, spectroscopy-style binary classification
problem (predict whether a target compound is present from noisy spectral
features). It is fully synthetic so the whole pipeline runs anywhere with a
fixed seed and no external downloads.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

FEATURE_PREFIX = "band_"
TARGET_COL = "target"


def make_spectral_dataset(
    n_samples: int = 2000,
    n_features: int = 24,
    n_informative: int = 8,
    class_sep: float = 1.2,
    weights: tuple[float, float] = (0.65, 0.35),
    random_state: int = 42,
) -> pd.DataFrame:
    """Return a reproducible synthetic classification dataset as a DataFrame.

    Columns are ``band_0 .. band_{n_features-1}`` plus an integer ``target``.
    The class imbalance (``weights``) mimics a realistic "compound present vs
    absent" detection task.
    """
    x, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=4,
        n_repeated=0,
        n_classes=2,
        weights=list(weights),
        class_sep=class_sep,
        flip_y=0.02,
        random_state=random_state,
    )
    columns = [f"{FEATURE_PREFIX}{i}" for i in range(n_features)]
    df = pd.DataFrame(x, columns=columns)
    df[TARGET_COL] = y.astype(int)
    return df


def feature_columns(df: pd.DataFrame, target_col: str = TARGET_COL) -> list[str]:
    return [c for c in df.columns if c != target_col]


def validate_dataset(df: pd.DataFrame, target_col: str = TARGET_COL) -> dict:
    """Run lightweight data-quality checks and return a structured report.

    Checks: row count, missing values, duplicate rows, non-numeric features,
    class balance, and a simple imbalance ratio. ``passed`` is True only when
    there are no missing values and every feature is numeric.
    """
    features = feature_columns(df, target_col)
    report: dict = {
        "n_rows": int(len(df)),
        "n_features": int(len(features)),
        "missing_values": int(df[features].isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
    }

    non_numeric = [c for c in features if not np.issubdtype(df[c].dtype, np.number)]
    report["non_numeric_features"] = non_numeric

    if target_col in df.columns:
        counts = {int(k): int(v) for k, v in df[target_col].value_counts().items()}
        report["class_counts"] = counts
        minority, majority = min(counts.values()), max(counts.values())
        report["imbalance_ratio"] = round(majority / minority, 3) if minority else None
    else:
        report["class_counts"] = {}
        report["imbalance_ratio"] = None

    report["passed"] = report["missing_values"] == 0 and len(non_numeric) == 0
    return report
