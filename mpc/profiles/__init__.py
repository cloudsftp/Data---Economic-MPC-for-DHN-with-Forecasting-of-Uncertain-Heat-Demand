from pathlib import Path
import pickle
import json
from typing import Union

import numpy as np
import pandas as pd


def get_standard_profile():
    with open("data/Profiles/dic_SLP_HEF_SF.pkl", "rb") as f:
        data = pickle.load(f)[3]["5 bis 10"]
        profile = np.zeros(24)
        for time in data:
            profile[time] = data[time]

        mean = profile.mean()
        profile = profile / mean

        return profile


class DemandForecaster:
    """
    Wrapper um ein gebündeltes LightGBM-24h-Modell (aus einem Pickle).

    Erwartet, dass im Pickle ein Dict liegt mit:
      - "models": list[lightgbm.LGBMRegressor]
      - "H": int
      - "params": dict (optional für Info)
      - "best_iters": list[int] (optional)
      - "featzres": features,
    """

    def __init__(self, models_path):
        self.models_path = Path(models_path)

        with open(self.models_path, "rb") as f:
            bundle = pickle.load(f)

        self.models = bundle["models"]
        self.H = bundle["H"]
        self.params = bundle.get("params", {})
        self.best_iters = bundle.get("best_iters", [None] * self.H)
        self.feature_names = bundle.get("featzres", None)

    def _prepare_X_for_lgbm(
        self, X: Union[pd.DataFrame, pd.Series, np.ndarray, list[float]]
    ) -> Union[pd.DataFrame, np.ndarray]:
        """
        Bereitet X so vor, dass LightGBM glücklich ist:
        - wenn DataFrame: Spaltenreihenfolge anpassen, numerische dtypes erzwingen
        - sonst: in 2D-numpy-array (N, num_features) casten
        """
        # Fall 1: DataFrame → mit Spaltennamen in der richtigen Reihenfolge
        if isinstance(X, pd.DataFrame):
            X_df = X.copy()

            if self.feature_names is not None:
                missing = [c for c in self.feature_names if c not in X_df.columns]
                if missing:
                    raise KeyError(f"Missing features for prediction: {missing}")
                X_df = X_df[self.feature_names]

            # numerische dtypes erzwingen
            X_df = X_df.apply(pd.to_numeric, errors="coerce")
            if X_df.isna().any().any():
                raise ValueError("NaNs in feature DataFrame after numeric conversion.")

            return X_df

        # Fall 2: Series → eine Zeile
        if isinstance(X, pd.Series):
            X_arr = X.to_numpy(dtype=float).reshape(1, -1)
            return X_arr

        # Fall 3: list / ndarray
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)

        return X_arr

    def predict(
        self, X: Union[pd.DataFrame, pd.Series, np.ndarray, list[float]]
    ) -> np.ndarray:
        """
        Vorhersage für ein Feature-Array X.
        Gibt ein Array mit Shape (N, H) zurück.
        """
        X_input = self._prepare_X_for_lgbm(X)

        preds = np.column_stack(
            [
                (
                    model.predict(X_input, num_iteration=best_it)
                    if best_it is not None
                    else model.predict(X_input)
                )
                for model, best_it in zip(self.models, self.best_iters)
            ]
        )
        return preds

    def predict_single(
        self, x_row: Union[pd.Series, pd.DataFrame, np.ndarray, list[float]]
    ) -> np.ndarray:
        """
        Prognose für genau einen Zeitpunkt (ein Feature-Vektor).
        Gibt einen Vektor der Länge H zurück (z.B. 24 Stunden).
        """
        # wenn eine Series/DF-Zeile kommt, lassen wir _prepare_X_for_lgbm entscheiden
        X_input = self._prepare_X_for_lgbm(x_row)
        preds = self.predict(X_input)  # (1, H)
        return preds[0]

    def predict_signal_json(
        self,
        x_row: Union[np.ndarray, pd.Series, list[float]],
        time_step: int = 60,
    ) -> str:
        """
        Liefert aus einem Feature-Vektor direkt ein JSON-Signal.
        """
        profile = self.predict_single(x_row)  # (H,)
        data = []
        t = 0
        for value in profile:
            data.append([t, float(value)])
            t += time_step

        return json.dumps(data, indent=2)
