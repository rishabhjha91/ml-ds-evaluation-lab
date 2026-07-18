# ML & DS Evaluation Lab

A small, fully reproducible lab that shows how I build **and evaluate** a
supervised machine-learning model end to end, serve it behind an API, and run
the kind of **data-science analytics** (funnel, cohort, A/B test, data-quality
checks) that back real product decisions.

Everything runs locally with a fixed seed and **no external data downloads** or
paid API keys. The dataset is synthetic (a spectroscopy-style "compound present
vs absent" detection task), so the whole pipeline is deterministic and easy to
review.

## What it demonstrates

| Area | Where |
| --- | --- |
| Machine learning (CV, model selection, calibration, error analysis) | `src/mldseval/train.py`, `evaluate.py` |
| Data validation / data quality | `src/mldseval/data.py`, `analytics.py` |
| Data science / SQL analytics (funnel, cohort, A/B test) | `src/mldseval/analytics.py` (DuckDB + SciPy) |
| Backend / MLOps (model serving) | `src/mldseval/api.py` (FastAPI) |
| Testing & CI | `tests/`, `.github/workflows/ci.yml` |
| Reproducibility & packaging | `pyproject.toml`, fixed seeds, pinned pipeline |

## Architecture

```
            make_spectral_dataset()        make_events()
                     |                            |
              validate_dataset()          DuckDB SQL analytics
                     |                     (funnel, cohort, A/B,
      StandardScaler + [LogReg | RandomForest]     data quality)
        (StratifiedKFold + GridSearchCV, ROC-AUC)
                     |
        held-out test: metrics, calibration,
        best-F1 threshold, error analysis
                     |
             artifacts/model.joblib  ->  FastAPI /predict
```

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# 1) Train + select a model, write artifacts/ (model + metrics.json)
python scripts/run_training.py        # or: mldseval-train

# 2) Run the data-science analytics suite
python scripts/run_analytics.py       # or: mldseval-analytics

# 3) Serve the trained model
uvicorn mldseval.api:app --reload
#   GET  /health
#   POST /predict   {"features": [ ... 24 floats ... ]}

# 4) Run the tests
pytest -q
ruff check .
```

## Design notes

- **No leakage:** scaling lives inside the pipeline, so it is fit only on
  training folds during cross-validation.
- **Model selection by cross-validated ROC-AUC** on the train split; the numbers
  reported in `artifacts/metrics.json` are on a **held-out test split**.
- **Calibration matters:** the Brier score and a reliability curve are reported,
  not just accuracy.
- **A/B testing** uses a proper two-proportion z-test (SciPy), not an informal
  comparison of two numbers.

## Repository layout

```
src/mldseval/     data, features, train, evaluate, api, analytics, cli
scripts/          run_training.py, run_analytics.py
tests/            data, evaluate, analytics, api tests
reports/          MODEL_CARD.md
```

## License

MIT — see `LICENSE`.
