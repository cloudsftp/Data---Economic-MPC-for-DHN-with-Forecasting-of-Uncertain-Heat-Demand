from pathlib import Path
from datetime import datetime, timedelta
from typing import cast

from loguru import logger
import pandas as pd

DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


CONSUMPTION_DATA_FILE = Path(
    "data/Profiles/consumer_cluster_data/aggregated_SFH_alt_5_1_features+lags.csv",
)


class Curve:
    current_hour: int
    hours_to_midnight: int

    air_temperature_today: float
    air_temperature_tomorrow: float

    def __init__(
        self,
        starting_hour: int,
        consumption_data_file: Path = CONSUMPTION_DATA_FILE,
    ):
        self.current_hour = starting_hour

        data = pd.read_csv(consumption_data_file)

        starting_date = datetime.strptime(
            str(data.loc[starting_hour, "date"]),
            DATE_FORMAT,
        )
        midnight = datetime.combine(
            starting_date.date() + timedelta(days=1),
            datetime.min.time(),
        )
        self.hours_to_midnight = int((midnight - starting_date).total_seconds() / 3600)

        logger.debug(
            f"current time: {starting_date}, hours to midnight: {self.hours_to_midnight}"
        )

        self.air_temperature_today = cast(
            float,
            data.loc[starting_hour, "temperature_day_mean"],
        )
        self.air_temperature_tomorrow = cast(
            float,
            data.loc[
                starting_hour + self.hours_to_midnight,
                "temperature_day_mean",
            ],
        )

        logger.debug(
            f"air temperature today: {self.air_temperature_today}, "
            f"tomorrow: {self.air_temperature_tomorrow}"
        )

    def control(self) -> list[float]:
        return self.hours_to_midnight * [
            Curve.feed_temperature(self.air_temperature_today)
        ] + (24 - self.hours_to_midnight) * [
            Curve.feed_temperature(self.air_temperature_tomorrow)
        ]

    @staticmethod
    def feed_temperature(air_temperature: float) -> float:
        return float(95 - 1.25 * air_temperature)
