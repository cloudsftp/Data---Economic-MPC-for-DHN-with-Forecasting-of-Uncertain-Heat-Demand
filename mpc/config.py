from pathlib import Path
from enum import Enum

from mpc.profiles.multi_consumer_forecast import (
    ClusterConfig,
)
from mpc.demand import DemandProvider, DemandProviderI, HistoricalDemandProvider

DEFAULT_NETWORK_DIRECTORY = Path("./data/aroma_mpc")


CONSUMER_DATA_DIRECOTRY: Path = Path("data/Profiles")
# Konfiguration: pro Verbraucher Feature- und Consumption-Datei + Modelldatei
CLUSTER_CONFIG: dict[str, ClusterConfig] = {
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


class FutureDataMode(Enum):
    Forecasted = "forecast"
    Actual = "actual"


def get_demand_provider(
    mode: FutureDataMode,
    base_hour: int,
    horizon_hours: int,
) -> DemandProviderI:
    num_points = horizon_hours + 1

    match mode:
        case FutureDataMode.Forecasted:
            demand_provider: DemandProviderI = DemandProvider(
                CONSUMER_DATA_DIRECOTRY,
                CLUSTER_CONFIG,
                base_hour,
                num_points,
            )
        case FutureDataMode.Actual:
            demand_provider = HistoricalDemandProvider(
                CONSUMER_DATA_DIRECOTRY,
                CLUSTER_CONFIG,
                base_hour,
                num_points,
            )

    return demand_provider
