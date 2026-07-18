"""FastAPI inference service.

Loads a trained model bundle (``artifacts/model.joblib``) lazily and exposes
``/health`` and ``/predict``. Train a model first with ``mldseval-train`` (or
``python scripts/run_training.py``).
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = Path(__file__).resolve().parents[2] / "artifacts" / "model.joblib"

app = FastAPI(title="ML-DS Evaluation Lab Inference API", version="0.1.0")

_model = None
_features: list[str] | None = None


class PredictRequest(BaseModel):
    features: list[float] = Field(..., description="Feature vector in training column order")


class PredictResponse(BaseModel):
    prediction: int
    probability: float


def load_model(path: Path = MODEL_PATH):
    """Load the model bundle from disk into module state."""
    global _model, _features
    bundle = joblib.load(path)
    _model = bundle["estimator"]
    _features = bundle["feature_names"]
    return _model


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": _model is not None}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    global _model, _features
    if _model is None:
        if MODEL_PATH.exists():
            load_model()
        else:
            raise HTTPException(
                status_code=503,
                detail="Model not available. Run 'mldseval-train' to create artifacts/model.joblib.",
            )
    if _features is not None and len(req.features) != len(_features):
        raise HTTPException(
            status_code=422,
            detail=f"Expected {len(_features)} features, got {len(req.features)}.",
        )
    x = np.asarray(req.features, dtype=float).reshape(1, -1)
    probability = float(_model.predict_proba(x)[0, 1])
    return PredictResponse(prediction=int(probability >= 0.5), probability=probability)
