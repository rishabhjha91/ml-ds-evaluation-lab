"""Command-line entrypoints for training and analytics."""

from __future__ import annotations

import json
from pathlib import Path

import joblib

from .analytics import run_all
from .data import make_spectral_dataset, validate_dataset
from .train import train_and_select


def _artifacts_dir() -> Path:
    root = Path(__file__).resolve().parents[2]
    art = root / "artifacts"
    art.mkdir(exist_ok=True)
    return art


def train_entrypoint() -> dict:
    """Generate data, validate, train + select a model, and persist artifacts."""
    art = _artifacts_dir()
    df = make_spectral_dataset()
    validation = validate_dataset(df)
    result = train_and_select(df)

    joblib.dump(
        {"estimator": result["estimator"], "feature_names": result["feature_names"]},
        art / "model.joblib",
    )
    summary = {
        "validation": validation,
        "best_model": result["best_model"],
        "cv_results": result["cv_results"],
        "test_metrics": result["test_metrics"],
        "best_threshold": result["best_threshold"],
        "error_analysis": result["error_analysis"],
    }
    (art / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary


def analytics_entrypoint() -> dict:
    """Run the SQL analytics suite and print the results as JSON."""
    results = run_all()
    print(json.dumps(results, indent=2, default=str))
    return results
