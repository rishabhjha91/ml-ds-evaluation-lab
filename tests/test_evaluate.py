import numpy as np
import pytest

from mldseval.evaluate import best_threshold, compute_metrics, error_analysis


def test_metrics_perfect_predictions():
    y_true = np.array([0, 1, 0, 1, 1])
    y_prob = np.array([0.1, 0.9, 0.2, 0.8, 0.95])
    y_pred = (y_prob >= 0.5).astype(int)
    m = compute_metrics(y_true, y_prob, y_pred)
    assert m["accuracy"] == 1.0
    assert m["f1"] == 1.0
    assert m["roc_auc"] == 1.0
    assert m["confusion_matrix"] == [[2, 0], [0, 3]]


def test_best_threshold_returns_valid_range():
    y_true = np.array([0, 0, 1, 1, 1, 0])
    y_prob = np.array([0.2, 0.4, 0.55, 0.6, 0.7, 0.35])
    out = best_threshold(y_true, y_prob)
    assert 0.05 <= out["threshold"] <= 0.95
    assert 0.0 <= out["f1"] <= 1.0


def test_error_analysis_finds_mistakes():
    y_true = np.array([0, 1, 0, 1])
    y_prob = np.array([0.95, 0.05, 0.1, 0.9])  # first two are confidently wrong
    cases = error_analysis(y_true, y_prob, top_k=5)
    assert len(cases) == 2
    assert cases[0]["index"] in (0, 1)




def test_error_analysis_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        error_analysis(np.array([0, 1]), np.array([0.2]), top_k=1)


def test_error_analysis_rejects_negative_top_k():
    with pytest.raises(ValueError, match="non-negative"):
        error_analysis(np.array([0, 1]), np.array([0.2, 0.8]), top_k=-1)


def test_error_analysis_returns_empty_for_zero_top_k():
    result = error_analysis(np.array([0, 1]), np.array([0.9, 0.1]), top_k=0)
    assert result == []
