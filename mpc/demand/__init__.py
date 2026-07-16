from typing import Protocol
from pathlib import Path

from mpc.profiles.multi_consumer_forecast import (
    MultiDemandForecaster,
    ClusterConfig,
)


def shorten_demands(demands: dict[str, list[float]], n: int) -> dict[str, list[float]]:
    for name in demands:
        demands[name] = demands[name][:n]

    return demands


class DemandProviderI(Protocol):
    def __init__(
        self,
        data_directory: Path,
        cluster_config: dict[str, ClusterConfig],
        base_hour: int,
        window: int,
    ) -> None: ...
    def get_previous_demands(self) -> dict[str, list[float]]: ...
    def get_actual_future_demands(self) -> dict[str, list[float]]: ...
    def get_predicted_future_demands(self) -> dict[str, list[float]]: ...
    def advance_step(self): ...
    def assert_enough_data_available(self, mpc_steps: int): ...


class DemandProvider(DemandProviderI):
    demand_forecaster: MultiDemandForecaster
    hour: int
    window: int

    def __init__(
        self,
        data_directory: Path,
        cluster_config: dict[str, ClusterConfig],
        base_hour: int,
        window: int,
    ):
        self.hour = base_hour
        self.window = window
        self.demand_forecaster = MultiDemandForecaster(
            data_directory,
            cluster_config,
        )

    def get_previous_demands(self) -> dict[str, list[float]]:
        return self.demand_forecaster.get_previous_real_data(
            self.hour,
            self.window,
        )

    def get_actual_future_demands(self) -> dict[str, list[float]]:
        return self.demand_forecaster.get_future_real_data(
            self.hour,
            self.window,
        )

    def get_predicted_future_demands(self) -> dict[str, list[float]]:
        actual_demand_start = self.demand_forecaster.get_future_real_data(
            self.hour,
            1,
        )

        predicted_demands = self.demand_forecaster.get_demands_for_hour(
            self.hour + 1,
        )
        predicted_demands = shorten_demands(
            predicted_demands,
            self.window - 1,
        )

        for key in predicted_demands:
            predicted_demands[key] = actual_demand_start[key] + predicted_demands[key]

        return predicted_demands

    def advance_step(self):
        self.hour += 1

    def assert_enough_data_available(self, mpc_steps: int):
        min_feature_length = min(
            len(df) for df in self.demand_forecaster.features.values()
        )
        required_length = self.hour + mpc_steps

        if required_length > min_feature_length:
            raise ValueError(
                f"MPC would run out of feature data!\n"
                f"Last required length of data = {required_length}\n"
                f"Feature length = {min_feature_length}\n"
                f"Reduce base_hour or mpc_steps."
            )


class HistoricalDemandProvider(DemandProviderI):
    """
    Demand provider that always returns the actual demand from historical data
    """

    demand_forecaster: MultiDemandForecaster
    hour: int
    window: int

    def __init__(
        self,
        data_directory: Path,
        cluster_config: dict[str, ClusterConfig],
        base_hour: int,
        window: int,
    ) -> None:
        self.hour = base_hour
        self.window = window
        self.demand_forecaster = MultiDemandForecaster(
            data_directory,
            cluster_config,
        )

    def get_previous_demands(self) -> dict[str, list[float]]:
        return self.demand_forecaster.get_previous_real_data(
            self.hour,
            self.window,
        )

    def get_actual_future_demands(self) -> dict[str, list[float]]:
        return self.demand_forecaster.get_future_real_data(
            self.hour,
            self.window,
        )

    def get_predicted_future_demands(self) -> dict[str, list[float]]:
        return self.demand_forecaster.get_future_real_data(
            self.hour,
            self.window,
        )

    def advance_step(self):
        self.hour += 1

    def assert_enough_data_available(self, mpc_steps: int):
        min_feature_length = min(
            len(df) for df in self.demand_forecaster.features.values()
        )
        required_length = self.hour + mpc_steps

        if required_length > min_feature_length:
            raise ValueError(
                f"MPC would run out of feature data!\n"
                f"Last required length of data = {required_length}\n"
                f"Feature length = {min_feature_length}\n"
                f"Reduce base_hour or mpc_steps."
            )
