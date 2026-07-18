# Model Card - ML & DS Evaluation Lab

This card summarises the model produced by `scripts/run_training.py`. All
numbers are reproducible (fixed seeds) and regenerated into
`artifacts/metrics.json` on every run; the values below are from a representative
run.

## Overview

- **Task:** binary classification (spectroscopy-style "compound present vs absent").
- **Data:** synthetic, 2,000 rows, 24 features, class balance ~1.8:1 (majority:minority).
- **Candidates:** logistic regression and random forest, each inside a
  `StandardScaler` pipeline.
- **Selection:** 5-fold stratified cross-validated ROC-AUC on the train split.
- **Reporting:** held-out 20% test split.

## Selected model

Random forest (`n_estimators=200`, `max_depth=None`), selected over logistic
regression on cross-validated ROC-AUC.

| Split | Metric | Value |
| --- | --- | --- |
| CV (train) | ROC-AUC (random forest) | 0.94 |
| CV (train) | ROC-AUC (logistic regression) | 0.82 |
| Test | ROC-AUC | 0.93 |
| Test | Accuracy | 0.88 |
| Test | Precision | 0.88 |
| Test | Recall | 0.75 |
| Test | F1 (threshold 0.5) | 0.81 |
| Test | F1 (best threshold 0.42) | 0.83 |
| Test | Brier score (calibration) | 0.11 |

Confusion matrix (test, threshold 0.5): `[[244, 14], [36, 106]]`.

## Threshold and calibration

The default 0.5 threshold under-detects the minority (positive) class. Grid
search over thresholds raises F1 from 0.81 to 0.83 at 0.42, trading a little
precision for recall. The Brier score (0.11) and reliability curve
(`evaluate.calibration_report`) quantify probability calibration.

## Error analysis

The most confident wrong predictions are surfaced by
`evaluate.error_analysis`; in the representative run they are all
false negatives (true positives the model scored with low probability),
consistent with the class imbalance.

## Limitations

- Synthetic data, so absolute numbers are illustrative; the point is the
  **methodology** (leakage-safe CV, calibration, threshold selection, error
  analysis), not the score.
- Two baseline model families only; no deep models or extensive tuning.
- No fairness/subgroup analysis (not meaningful on synthetic features).

## Intended use

A reference example of a disciplined, reproducible ML workflow and of how I
evaluate models. Not intended for any real-world decision-making.
