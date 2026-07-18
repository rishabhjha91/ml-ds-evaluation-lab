"""Evaluation utilities: metrics, threshold selection, calibration, error analysis."""

from __future__ import annotations

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(y_true, y_prob, y_pred) -> dict:
    """Return the standard classification metrics plus a calibration score."""
    y_true = np.asarray(y_true)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def best_threshold(y_true, y_prob) -> dict:
    """Grid-search the probability threshold that maximises F1."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    thresholds = np.linspace(0.05, 0.95, 91)
    best_t, best_f1 = 0.5, -1.0
    for t in thresholds:
        pred = (y_prob >= t).astype(int)
        score = f1_score(y_true, pred, zero_division=0)
        if score > best_f1:
            best_t, best_f1 = float(t), float(score)
    return {"threshold": round(best_t, 3), "f1": round(best_f1, 4)}


def calibration_report(y_true, y_prob, n_bins: int = 10) -> dict:
    """Return reliability-curve points and the Brier score."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="quantile")
    return {
        "brier": float(brier_score_loss(y_true, y_prob)),
        "mean_predicted": [float(v) for v in mean_pred],
        "fraction_positive": [float(v) for v in frac_pos],
    }


def error_analysis(y_true, y_prob, top_k: int = 10) -> list[dict]:
    """Return the most confident *wrong* predictions for manual inspection."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= 0.5).astype(int)
    wrong = np.where(y_pred != y_true)[0]
    if wrong.size == 0:
        return []
    confidence = np.abs(y_prob - 0.5)
    order = wrong[np.argsort(-confidence[wrong])][:top_k]
    return [
        {"index": int(i), "y_true": int(y_true[i]), "y_prob": round(float(y_prob[i]), 4)}
        for i in order
    ]
