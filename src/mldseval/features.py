"""Feature preprocessing.

Kept intentionally small and leakage-safe: scaling lives inside the model
pipeline so it is always fit on training folds only (never on validation or
test data).
"""

from __future__ import annotations

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_preprocessor() -> Pipeline:
    """Return a preprocessing pipeline (standardisation of numeric features)."""
    return Pipeline(steps=[("scaler", StandardScaler())])
