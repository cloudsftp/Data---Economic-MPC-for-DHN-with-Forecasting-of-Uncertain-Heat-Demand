from typing import TypeVar
import math

import pandas as pd

T2 = 59.2453
T1 = 220.536
T0 = 1.93729


T = TypeVar("T", float, pd.DataFrame, pd.Series)


def temperature(energy_density: T) -> T:
    energy_density /= 1e9
    return (T2 * energy_density + T1) * energy_density + T0


def energy_density(temperature: float) -> float:
    d = (T1**2) - 4 * T2 * (T0 - temperature)
    return 1e9 * (-T1 + math.sqrt(d)) / (2 * T2)


MIN_TEMPERATURE = 70.0
MIN_TEMPERATURE_W_BUFFER = 70.25

MIN_ENERGY_DENSITY = energy_density(MIN_TEMPERATURE)
MIN_ENERGY_DENSITY_W_BUFFER = energy_density(MIN_TEMPERATURE_W_BUFFER)
