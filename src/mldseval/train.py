"""Training and model selection.

Uses a leakage-safe pipeline (scaler + classifier), stratified k-fold
cross-validation, and a small grid search over two baselines (logistic
regression and random forest). Model selection is by cross-validated ROC-AUC
on the train split; final numbers are reported on a held-out test split.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data import TARGET_COL, feature_columns
from .evaluate import best_threshold, compute_metrics, error_analysis

_MODEL_GRID = {
    "logreg": (
        LogisticRegression(max_iter=2000),
        {"clf__C": [0.1, 1.0, 10.0]},
    ),
    "random_forest": (
        RandomForestClassifier(random_state=0, n_jobs=-1),
        {"clf__n_estimators": [200], "clf__max_depth": [None, 8]},
    ),
}


def _pipeline(clf) -> Pipeline:
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])


def train_and_select(
    df,
    target_col: str = TARGET_COL,
    test_size: float = 0.2,
    n_splits: int = 5,
    random_state: int = 42,
) -> dict:
    """Train baselines, select the best by CV ROC-AUC, and evaluate on test.

    Returns a dict with the fitted estimator, per-model CV results, held-out
    test metrics, best decision threshold, and a small error-analysis sample.
    """
    features = feature_columns(df, target_col)
    x = df[features].to_numpy()
    y = df[target_col].to_numpy()

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, stratify=y, random_state=random_state
    )
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    cv_results: dict = {}
    best_name, best_est, best_score = None, None, -np.inf
    for name, (clf, grid) in _MODEL_GRID.items():
        search = GridSearchCV(_pipeline(clf), grid, scoring="roc_auc", cv=cv, n_jobs=-1)
        search.fit(x_train, y_train)
        cv_results[name] = {
            "best_params": search.best_params_,
            "cv_roc_auc": round(float(search.best_score_), 4),
        }
        if search.best_score_ > best_score:
            best_name, best_est, best_score = name, search.best_estimator_, search.best_score_

    y_prob = best_est.predict_proba(x_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    return {
        "best_model": best_name,
        "cv_results": cv_results,
        "test_metrics": compute_metrics(y_test, y_prob, y_pred),
        "best_threshold": best_threshold(y_test, y_prob),
        "error_analysis": error_analysis(y_test, y_prob, top_k=5),
        "estimator": best_est,
        "feature_names": features,
    }
