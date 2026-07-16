import json
import pickle

import numpy as np
import pandas as pd

from mpc.profiles import DemandForecaster
from mpc.profiles.multi_consumer_forecast import ClusterConfig, MultiDemandForecaster


class DummyModel:
    """
    Minimales Fake-Model mit predict(X):
    gibt sum(X_row) + h zurück.
    """

    def __init__(self, h: int):
        self.h = h

    def predict(self, X, hour=None):
        # X: (N, num_features)
        return X.sum(axis=1) + self.h


def _create_model_bundle(tmp_path):
    H = 3  # kleiner Test-Horizont

    bundle = {
        "models": [DummyModel(h=i) for i in range(H)],
        "H": H,
        "params": {"test_param": 123},
        "best_iters": [None] * H,
        "featzres": ["a", "b"],
    }

    file_path = tmp_path / "lightgbm_models_test.pkl"
    with open(file_path, "wb") as f:
        pickle.dump(bundle, f)

    return file_path


def test_forecaster_loading(tmp_path):
    bundle_path = _create_model_bundle(tmp_path)

    forecaster = DemandForecaster(models_path=str(bundle_path))

    # Anzahl Modelle und H sollten konsistent sein
    assert forecaster.H == 3
    assert len(forecaster.models) == 3
    # params und best_iters sollten übernommen worden sein
    assert isinstance(forecaster.params, dict)
    assert forecaster.params["test_param"] == 123
    assert forecaster.best_iters == [None, None, None]


def test_forecaster_predict_numpy(tmp_path):
    bundle_path = _create_model_bundle(tmp_path)
    forecaster = DemandForecaster(models_path=str(bundle_path))

    X = np.array([[1.0, 2.0], [3.0, 4.0]])  # Shape (2, 2)

    preds = forecaster.predict(X)

    # Erwartete Shape: (N, H) = (2, 3)
    assert preds.shape == (2, 3)

    expected = np.column_stack([X.sum(axis=1) + h for h in range(forecaster.H)])

    assert np.array_equal(preds, expected)


def test_forecaster_predict_single_numpy(tmp_path):
    bundle_path = _create_model_bundle(tmp_path)
    forecaster = DemandForecaster(models_path=str(bundle_path))

    x_row = np.array([1.0, 2.0])  # Shape (2,)

    preds_single = forecaster.predict_single(x_row)

    # Shape (H,)
    assert preds_single.shape == (forecaster.H,)

    # Erwartung: sum(x_row) + h
    expected = np.array([x_row.sum() + h for h in range(forecaster.H)])
    assert np.array_equal(preds_single, expected)


def test_forecaster_predict_signal_json(tmp_path):
    bundle_path = _create_model_bundle(tmp_path)
    forecaster = DemandForecaster(models_path=str(bundle_path))

    x_row = np.array([1.0, 2.0])

    signal_json = forecaster.predict_signal_json(x_row, time_step=60)

    data = json.loads(signal_json)

    # Länge muss H sein
    assert len(data) == forecaster.H

    # Struktur: [[t0, v0], [t1, v1], ...]
    # t = k * 60
    for k, (t, v) in enumerate(data):
        assert t == k * 60

    # Werte sollten zu predict_single passen
    expected_profile = forecaster.predict_single(x_row)
    extracted_values = np.array([v for _, v in data])
    assert np.allclose(extracted_values, expected_profile)


def test_multi_forecaster_loading(tmp_path):
    # Dummy-Model-Bundle anlegen
    models_file = _create_model_bundle(tmp_path).name

    # Künstliche Consumer-Config für den Test
    consumer_config = {
        "C1": ClusterConfig(
            models_file_name=models_file,
            data_file_name="X_features_C1.csv",
        ),
        "C2": ClusterConfig(
            models_file_name=models_file,
            data_file_name="X_features_C2.csv",
        ),
    }

    # Feature-Dateien für C1 und C2 erzeugen
    data_dir = tmp_path / "consumer_cluster_data"
    data_dir.mkdir(parents=True, exist_ok=True)

    df_C1 = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=2, freq="h"),
            "a": [1.0, 2.0],
            "b": [3.0, 4.0],
            "consumption_kWh": [10.0, 11.0],
        }
    )
    df_C2 = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=2, freq="h"),
            "a": [5.0, 6.0],
            "b": [7.0, 8.0],
            "consumption_kWh": [20.0, 21.0],
        }
    )

    df_C1.to_csv(data_dir / "X_features_C1.csv", index=False)
    df_C2.to_csv(data_dir / "X_features_C2.csv", index=False)

    # models/ Ordner und Bundle-Datei an die richtige Stelle legen
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    (tmp_path / models_file).replace(models_dir / models_file)

    # MultiDemandForecaster
    forecaster = MultiDemandForecaster(
        base_dir=tmp_path,
        cluster_config=consumer_config,
    )

    # Prüfen, ob Forecaster und Features für beide Consumer geladen wurden
    assert set(forecaster.forecasters.keys()) == {"C1", "C2"}
    assert set(forecaster.features.keys()) == {"C1", "C2"}

    # H aus einem der Single-Forecaster checken
    assert forecaster.forecasters["C1"].H == 3


def test_multi_forecaster_get_demands_for_iteration(tmp_path):
    models_file = _create_model_bundle(tmp_path).name

    consumer_config = {
        "C1": ClusterConfig(
            models_file_name=models_file,
            data_file_name="X_features_C1.csv",
        ),
        "C2": ClusterConfig(
            models_file_name=models_file,
            data_file_name="X_features_C2.csv",
        ),
    }

    data_dir = tmp_path / "consumer_cluster_data"
    data_dir.mkdir(parents=True, exist_ok=True)

    df_C1 = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=1, freq="h"),
            "a": [1.0],
            "b": [2.0],
            "consumption_kWh": [3.0],  # beliebiger Wert, wird im Test nicht verwendet
        }
    )
    df_C2 = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=1, freq="h"),
            "a": [10.0],
            "b": [20.0],  # sum = 30
            "consumption_kWh": [30.0],
        }
    )

    df_C1.to_csv(data_dir / "X_features_C1.csv", index=False)
    df_C2.to_csv(data_dir / "X_features_C2.csv", index=False)

    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    (tmp_path / models_file).replace(models_dir / models_file)

    forecaster = MultiDemandForecaster(
        base_dir=tmp_path,
        cluster_config=consumer_config,
    )

    demands = forecaster.get_demands_for_hour(0)

    # Beide Consumer sollten im Dict sein
    assert set(demands.keys()) == {"C1", "C2"}

    # Länge jedes Profils = H
    H = forecaster.forecasters["C1"].H
    assert len(demands["C1"]) == H
    assert len(demands["C2"]) == H

    # Erwartete Werte:
    # DummyModel: sum(x_row) + h
    expected_C1 = np.array([3.0 + h for h in range(H)])  # sum(1,2) = 3
    expected_C2 = np.array([30.0 + h for h in range(H)])  # sum(10,20) = 30

    assert np.allclose(demands["C1"], expected_C1)
    assert np.allclose(demands["C2"], expected_C2)
