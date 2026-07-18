import numpy as np

from mldseval.data import TARGET_COL, make_spectral_dataset, validate_dataset


def test_dataset_shape_and_target():
    df = make_spectral_dataset(n_samples=500, n_features=16)
    assert len(df) == 500
    assert TARGET_COL in df.columns
    assert df.shape[1] == 17  # 16 features + target
    assert set(df[TARGET_COL].unique()).issubset({0, 1})


def test_dataset_is_reproducible():
    a = make_spectral_dataset(n_samples=200, random_state=123)
    b = make_spectral_dataset(n_samples=200, random_state=123)
    assert a.equals(b)


def test_validation_passes_on_clean_data():
    df = make_spectral_dataset(n_samples=400)
    report = validate_dataset(df)
    assert report["passed"] is True
    assert report["missing_values"] == 0
    assert report["imbalance_ratio"] >= 1.0


def test_validation_flags_missing_values():
    df = make_spectral_dataset(n_samples=400)
    df.loc[0, "band_0"] = np.nan
    report = validate_dataset(df)
    assert report["missing_values"] == 1
    assert report["passed"] is False
