from dataclasses import dataclass
from pathlib import Path
from loguru import logger
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from mpc.profiles import DemandForecaster


@dataclass
class ClusterConfig:
    data_file_name: str
    models_file_name: str


class MultiDemandForecaster:
    """
    Verwaltet mehrere DemandForecaster (für C1..C5).
    Lädt:
      - Modelle
      - Feature-Tabellen mit Lags (Input für das Modell)
    """

    base_dir: Path
    cluster_config: dict[str, ClusterConfig]

    forecasters: dict[str, DemandForecaster] = {}
    features: dict[str, pd.DataFrame] = {}
    consumptions: dict[str, pd.DataFrame] = {}

    def __init__(
        self,
        base_dir: Path,
        cluster_config: dict[str, ClusterConfig],
    ):
        self.base_dir = base_dir
        self.cluster_config = cluster_config

        for consumer, cfg in self.cluster_config.items():
            models_path = self.base_dir / "models" / cfg.models_file_name
            feat_path = self.base_dir / "consumer_cluster_data" / cfg.data_file_name

            logger.info(f"[LOAD] {consumer}:")
            logger.info("  Models:", models_path)
            logger.info("  Feature+Lag data:", feat_path)

            # 1) Modell laden
            forecaster = DemandForecaster(models_path)
            self.forecasters[consumer] = forecaster

            # 2) Features+Consumption gemeinsam laden
            df_feat = pd.read_csv(feat_path)

            # Datumsfeld konvertieren
            if "date" not in df_feat.columns:
                raise KeyError(f"'date' fehlt in Feature-Datei für {consumer}.")
            df_feat["date"] = pd.to_datetime(df_feat["date"], utc=False, format="mixed")

            # Robuste numerische Konvertierung
            df_feat = df_feat.apply(pd.to_numeric, errors="coerce")

            # Zeilen mit NA entfernen
            df_feat = df_feat.dropna().reset_index(drop=True)

            # 3) Consumption separat speichern
            if "consumption_kWh" not in df_feat.columns:
                raise KeyError(
                    f"'consumption_kWh' fehlt in Feature-Datei für {consumer}."
                )

            self.consumptions[consumer] = df_feat[["date", "consumption_kWh"]].copy()

            # 4) Feature-DF: exakt die Spalten, die das Modell erwartet
            if forecaster.feature_names is None:
                # Fallback: alles außer date und target und evtl. y_t+...
                drop_cols = [c for c in df_feat.columns if c.startswith("y_t+")]
                feature_df = df_feat.drop(columns=drop_cols, errors="ignore")
                logger.warning(
                    "forecaster.feature_names is None, using heuristic feature selection."
                )
            else:
                missing = [
                    c for c in forecaster.feature_names if c not in df_feat.columns
                ]
                if missing:
                    raise KeyError(
                        f"Feature columns missing in file for {consumer}: {missing}"
                    )
                feature_df = df_feat[forecaster.feature_names].copy()

            self.features[consumer] = feature_df

    def get_demands_for_hour(self, iteration: int) -> dict[str, list[float]]:
        """
        Liefert 24h-Profile für alle Verbraucher, basierend auf Zeile `iteration`
        im jeweiligen Feature-DF (mit Lags).
        """
        demands: dict[str, list[float]] = {}

        for consumer, forecaster in self.forecasters.items():
            df_feat = self.features[consumer]

            if iteration >= len(df_feat):
                raise IndexError(
                    f"Iteration {iteration} außerhalb des Feature-Bereichs für {consumer}"
                )

            x_row = df_feat.iloc[iteration]
            profile = forecaster.predict_single(x_row)  # np.ndarray (H,)
            demands[consumer] = profile.tolist()

        return demands

    def _get_real_data(
        self,
        start: int,
        end: int,
    ):
        window = end - start

        # --- extract real data ---
        demands = {}
        for consumer, df_cons in self.consumptions.items():
            hist_slice = df_cons.iloc[start:end]

            values = hist_slice["consumption_kWh"].to_numpy().tolist()

            if len(values) != window:
                raise ValueError(
                    f"Not enough data for {consumer}: expected {window}, have {len(values)}"
                )

            demands[consumer] = values

        return demands

    def get_previous_real_data(
        self,
        hour: int,
        window: int = 25,
    ):
        """
        Returns the last `window` real consumption values (consumption_kWh)
        for each consumer prior to hour.
        """
        hour = hour + 1
        return self._get_real_data(hour - window, hour)

    def get_future_real_data(
        self,
        hour: int,
        window: int = 25,
    ) -> dict[str, list[float]]:
        """
        Returns the next `window` real consumption values (consumption_kWh)
        for each consumer after hour.
        """
        return self._get_real_data(hour, hour + window)


def mape(y_true, y_pred) -> float:
    """MAPE in %, mit Schutz vor Division durch 0."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    eps = 1e-9

    return float(
        np.mean(np.abs(y_true - y_pred) / np.clip(np.abs(y_true), eps, None)) * 100.0
    )


def pick_test_base_iters(
    forecast_all,
    H: int,
    max_shift: int,
    ref_consumer: str = "C1",
    n_examples: int = 3,
    test_frac: float = 0.2,
) -> list[int]:
    """
    Wählt base_iter aus dem Testbereich (letzte 20% der Daten),
    so dass für ALLE Consumer gilt:
      - 24h-True-Fenster [base_iter+1 .. base_iter+H] existiert
      - Features für Forecast-Start bei base_iter + max_shift existieren.

    Gibt n_examples Basis-Indizes zurück, ungefähr gleichmäßig verteilt.
    """

    # Referenzserie nur für Datum / Debug
    df_ref = forecast_all.consumptions[ref_consumer].copy()
    df_ref["date"] = pd.to_datetime(df_ref["date"], format="mixed", utc=False)
    df_ref = df_ref.sort_values("date").reset_index(drop=True)
    N_ref = len(df_ref)

    # Globale Grenzen über ALLE Consumer
    min_len_feat = min(len(df) for df in forecast_all.features.values())
    min_len_cons = min(len(df) for df in forecast_all.consumptions.values())

    # 24h-True-Fenster für alle Consumer: base_iter+1+H <= min_len_cons
    upper_bound_cons = min_len_cons - 1 - H

    # Features für Forecast bei base_iter + max_shift:
    upper_bound_feat = min_len_feat - 1 - max_shift

    global_upper = min(upper_bound_cons, upper_bound_feat, N_ref - 1)
    if global_upper <= 0:
        raise ValueError(
            f"global_upper={global_upper} zu klein; "
            f"prüfe H={H}, max_shift={max_shift}, Datenlängen."
        )

    # Testbereich in der Referenzserie
    test_start = int((1.0 - test_frac) * N_ref)
    test_end = global_upper
    if test_end <= test_start:
        raise ValueError(
            f"Testbereich leer oder zu klein: "
            f"test_start={test_start}, test_end={test_end}, "
            f"global_upper={global_upper}, N_ref={N_ref}"
        )

    candidate_indices = list(range(test_start, test_end + 1))

    valid: list[int] = []
    for idx in candidate_indices:
        # 1) True-Fenster für alle Consumer?
        if idx + 1 + H > min_len_cons:
            continue

        # 2) Features für alle Forecast-Offets (bis max_shift)?
        if idx + max_shift > min_len_feat - 1:
            continue

        valid.append(idx)

    if not valid:
        raise ValueError(
            "Kein gültiger base_iter im Testbereich gefunden – "
            "prüfe H/max_shift und Zeitreihenlängen."
        )

    # n_examples ungefähr gleichmäßig aus valid ziehen
    step = max(len(valid) // (n_examples + 1), 1)
    picks = [valid[i * step] for i in range(1, n_examples + 1)]

    logger.debug("Gewählte Test-base_iter (mit Datum):")
    for b in picks:
        dt = df_ref.loc[b, "date"]
        logger.debug(f"  base_iter={b}, date={dt}")

    return picks


def plot_multi_shift_forecasts_multi(
    forecast_all,
    base_iter: int,
    shifts=[0, 3, 6, 12, 18],
):
    """
    Zeichnet für C1–C4 jeweils einen Subplot:
      - Reale Last: exakt 24h ab base_iter (Stunde 0)
      - Forecast-Kurven ab base_iter + shift
        (nur der Teil im 24h-Fenster wird geplottet und für MAPE genutzt)
    """
    any_forecaster = next(iter(forecast_all.forecasters.values()))
    H = any_forecaster.H  # z.B. 24
    horizon = 24
    max_shift = max(shifts)

    # --- Fraunhofer-Farbpalette ---
    fraunhofer_colors = {
        "fraunhofer_green": "#179c7d",  # Primär
        "steel_blue": "#005b7f",
        "silver_grey": "#a6bbc8",
        "orange": "#f58220",
        "graphit": "#1c3f52",
        "sand": "#d3c7ae",
        "petrol": "#008598",
        "aqua": "#39c1cd",
        "lime": "#b2d235",
        "gelb": "#fdb913",
        "rot": "#bb0056",
        "weinrot": "#7c154d",
    }

    offset_colors = {
        0: fraunhofer_colors["fraunhofer_green"],
        3: fraunhofer_colors["sand"],
        6: fraunhofer_colors["orange"],
        12: fraunhofer_colors["lime"],
        18: fraunhofer_colors["rot"],
    }
    fallback_colors = list(fraunhofer_colors.values())

    min_len_feat = min(len(df) for df in forecast_all.features.values())
    min_len_cons = min(len(df) for df in forecast_all.consumptions.values())

    if base_iter + max_shift > min_len_feat - 1:
        raise ValueError(
            f"base_iter={base_iter} zu groß für Features: "
            f"max_shift={max_shift}, min_len_feat={min_len_feat}"
        )
    if base_iter + 1 + horizon > min_len_cons:
        raise ValueError(
            f"base_iter={base_iter} zu groß für Consumption: "
            f"horizon={horizon}, min_len_cons={min_len_cons}"
        )

    consumers_to_plot = ["C1", "C2", "C3", "C4"]

    # Forecasts für alle Offsets vorab berechnen
    demands_by_offset = {}
    for offset in shifts:
        iter_forecast = base_iter + offset
        logger.info(
            f"Berechne Forecasts für iteration={iter_forecast} (offset={offset}h)"
        )
        demands_by_offset[offset] = forecast_all.get_demands_for_iteration(
            iter_forecast
        )

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=False)
    axes = axes.ravel()

    markers = ["o", "s", "^", "D", "x"]

    for ax, consumer in zip(axes, consumers_to_plot):
        df_cons = forecast_all.consumptions[consumer].copy()

        if "date" not in df_cons.columns or "consumption_kWh" not in df_cons.columns:
            raise KeyError(
                f"'date' oder 'consumption_kWh' fehlen im Consumption-DF von {consumer}."
            )

        df_cons["date"] = pd.to_datetime(df_cons["date"], format="mixed", utc=False)

        # 24h-Realfenster: ab base_iter + 1 (t+1) bis t+24
        real_start_idx = base_iter + 1
        real_end_idx = real_start_idx + horizon

        times_real = df_cons["date"].iloc[real_start_idx:real_end_idx]
        vals_real = (
            df_cons["consumption_kWh"].iloc[real_start_idx:real_end_idx].to_numpy()
        )

        # True
        ax.plot(
            times_real,
            vals_real,
            label="Real Consumption",
            color=fraunhofer_colors["graphit"],
            linewidth=2.0,
        )

        # Prognosen für alle Offsets, nur im 24h-Fenster
        for i, offset in enumerate(shifts):
            color = offset_colors.get(
                offset,
                fallback_colors[i % len(fallback_colors)],
            )
            marker = markers[i % len(markers)]

            iter_forecast = base_iter + offset
            full_profile = np.array(demands_by_offset[offset][consumer])

            pred_start_idx = iter_forecast + 1
            pred_end_idx = pred_start_idx + H

            ov_start = max(pred_start_idx, real_start_idx)
            ov_end = min(pred_end_idx, real_end_idx)

            if ov_end <= ov_start:
                continue

            true_segment = df_cons["consumption_kWh"].iloc[ov_start:ov_end].to_numpy()
            times_segment = df_cons["date"].iloc[ov_start:ov_end]

            seg_len = len(true_segment)
            if seg_len <= 0:
                continue

            idx0 = ov_start - pred_start_idx
            idx1 = idx0 + seg_len
            pred_segment = full_profile[idx0:idx1]

            m = mape(true_segment, pred_segment)
            label = f"+{offset}h (MAPE={m:.1f}%)"

            ax.plot(
                times_segment,
                pred_segment,
                linestyle="--",
                marker=marker,
                color=color,
                linewidth=1.5,
                label=label,
            )

        ax.set_title(f"Cluster {consumer[1:]}")
        ax.set_ylabel("Consumption (scaled)")
        ax.grid(True)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m %H:%M"))
        ax.tick_params(axis="x", rotation=45)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")

    # fig.suptitle(
    #     f"Real Consumption and Consumption Forecast from various starting points \n"
    #     f"(base_iter={base_iter}, 24h-Fenster)",
    #     fontsize=14,
    # )
    plt.tight_layout(rect=(0, 0, 0.9, 0.95))
    plt.show()


if __name__ == "__main__":
    print("== Erzeuge MultiDemandForecaster ==")
    consumer_data_directory = Path("data/Profiles")
    cluster_config = {
        "C1": ClusterConfig(
            data_file_name="aggregated_SFH_alt_5_1_features+lags.csv",
            models_file_name="lightgbm_models_aggregated_SFH_alt_5_1_features+lags.pkl",
        ),
        "C2": ClusterConfig(
            data_file_name="aggregated_SFH_alt_5_2_features+lags.csv",
            models_file_name="lightgbm_models_aggregated_SFH_alt_5_2_features+lags.pkl",
        ),
        "C3": ClusterConfig(
            data_file_name="aggregated_SFH_alt_5_3_features+lags.csv",
            models_file_name="lightgbm_models_aggregated_SFH_alt_5_3_features+lags.pkl",
        ),
        "C4": ClusterConfig(
            data_file_name="aggregated_SFH_alt_5_4_features+lags.csv",
            models_file_name="lightgbm_models_aggregated_SFH_alt_5_4_features+lags.pkl",
        ),
        "C5": ClusterConfig(
            data_file_name="aggregated_SFH_alt_5_5_features+lags.csv",
            models_file_name="lightgbm_models_aggregated_SFH_alt_5_5_features+lags.pkl",
        ),
    }
    forecast_all = MultiDemandForecaster(consumer_data_directory, cluster_config)

    any_forecaster = next(iter(forecast_all.forecasters.values()))
    H = any_forecaster.H
    shifts = [0, 3, 6, 12, 18]
    max_shift = max(shifts)

    # ---------------- Beispieltage aus Testbereich ----------------
    test_base_iters = pick_test_base_iters(
        forecast_all,
        H=H,
        max_shift=max_shift,
        ref_consumer="C2",
        n_examples=6,  # z.B. drei zusätzliche Tage
    )
    print(test_base_iters)

    for k, base_iter in enumerate(test_base_iters, start=1):
        plot_multi_shift_forecasts_multi(
            forecast_all,
            base_iter=base_iter,
            shifts=shifts,
        )
