from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mldseval import api
from mldseval.data import feature_columns, make_spectral_dataset


def _fit_small_model():
    df = make_spectral_dataset(n_samples=400)
    features = feature_columns(df)
    model = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000))])
    model.fit(df[features].to_numpy(), df["target"].to_numpy())
    return model, features, df


def test_health_endpoint():
    client = TestClient(api.app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_predict_endpoint():
    model, features, df = _fit_small_model()
    api._model = model
    api._features = features
    client = TestClient(api.app)

    payload = {"features": df[features].iloc[0].tolist()}
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert 0.0 <= body["probability"] <= 1.0
    assert body["prediction"] in (0, 1)


def test_predict_rejects_wrong_feature_count():
    model, features, _ = _fit_small_model()
    api._model = model
    api._features = features
    client = TestClient(api.app)

    resp = client.post("/predict", json={"features": [0.1, 0.2, 0.3]})
    assert resp.status_code == 422
