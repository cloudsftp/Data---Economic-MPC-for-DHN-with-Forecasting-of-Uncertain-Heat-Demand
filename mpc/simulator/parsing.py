from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd
from loguru import logger
from scipy import integrate

from mpc.signals import rotate_signal
from mpc.water import energy_density


@dataclass
class ResultEvaluation:
    objective: np.float64
    gradients: np.ndarray

    min_temperature: np.ndarray
    min_temperature_gradients: np.ndarray

    min_temperature_violation: np.float64

    heat_fed_in: np.float64
    heat_delivered: np.float64
    heat_loss: np.float64
    pump_energy: np.float64

    result: pd.DataFrame


def _read_output_csv(result_file_path: Path) -> pd.DataFrame:
    return pd.read_csv(
        result_file_path,
        skiprows=[1],
        index_col="time",
    )


def count_consumers(result_file_path: Path) -> int:
    outputs = _read_output_csv(result_file_path)

    consumer_variables = list(
        filter(
            lambda v: v.startswith("C") and v.endswith("0.e"), map(str, outputs.columns)
        )
    )

    return len(consumer_variables)


def evaluate_result_file(
    result_file_path: Path,
    num_time_steps: int,
    num_fixed_control: int,
    num_relevant_control: int,
    iteration: int,
    num_skip_temp_constaints: int,
    min_temperature: float,
    lowest_temperature_consumer: str,
) -> ResultEvaluation:
    logger.debug(
        f"computing the objective function and gradients for result {result_file_path}"
    )
    outputs = _read_output_csv(result_file_path)

    return evaluate_results(
        outputs,
        num_time_steps,
        num_fixed_control,
        num_relevant_control,
        iteration,
        num_skip_temp_constaints,
        min_temperature,
        lowest_temperature_consumer,
    )


def evaluate_results(
    outputs: pd.DataFrame,
    num_time_steps: int,
    num_fixed_control: int,
    num_relevant_control: int,
    iteration: int,
    num_skip_temp_constaints: int,
    min_temperature: float,
    lowest_temperature_consumer: str,
) -> ResultEvaluation:
    outputs = outputs.loc[outputs.index >= 0.0]

    if len(outputs) > num_time_steps:
        logger.warning(
            f"Simulation result has {len(outputs)} rows, only evaluating first {num_time_steps}"
        )
        outputs = outputs.iloc[:num_time_steps]

    columns = list(map(str, outputs.columns))

    consumer_variables = list(
        filter(lambda v: v.startswith("C") and v.endswith("0.e"), columns)
    )
    logger.trace(f"num consumer variables: {len(consumer_variables)}")
    min_temperature_consumer_variable = f"{lowest_temperature_consumer}.0.e"

    consumer_power_variables = list(
        filter(lambda v: v.startswith("C") and v.endswith("Q"), columns)
    )

    source_variables = list(
        filter(
            lambda v: v.startswith("S") and v.endswith("Q"),
            columns,
        )
    )
    source_names = list(
        set(
            map(
                lambda v: v.split(".")[0],
                filter(
                    lambda v: v.startswith("S"),
                    columns,
                ),
            )
        )
    )

    controls = list(
        set(
            map(
                lambda v: v.split("|")[-1],
                filter(lambda v: "|" in v, columns),
            )
        )
    )
    controls.sort(key=lambda v: int(v.split("_")[-1]))
    controls = rotate_signal(controls, -iteration)
    controls = controls[num_fixed_control : num_relevant_control + num_fixed_control]
    logger.trace(f"controls: {controls}")

    min_energy_density = energy_density(min_temperature)

    heat_fed_in = _compute_energy_fed_in(
        outputs,
        source_variables,
    )
    heat_delivered = _compute_energy_delivered(
        outputs,
        consumer_power_variables,
    )
    heat_loss = heat_fed_in - heat_delivered
    pump_energy = _compute_pump_energy(
        outputs,
        source_names,
    )

    return ResultEvaluation(
        _compute_objective(outputs, source_variables, consumer_power_variables),
        _compute_gradients(
            outputs,
            source_variables,
            consumer_power_variables,
            controls,
        ),
        _compute_min_temperature(
            outputs,
            consumer_variables,
            num_skip_temp_constaints,
            min_energy_density,
        ),
        _compute_min_temperature_gradients(
            outputs,
            consumer_variables,
            controls,
            num_skip_temp_constaints,
        ),
        _compute_min_temperature_violation(
            outputs,
            min_temperature_consumer_variable,
            min_energy_density,
        ),
        heat_fed_in,
        heat_delivered,
        heat_loss,
        pump_energy,
        outputs,
    )


def _compute_objective(
    outputs: pd.DataFrame,
    source_variables: list[str],
    consumer_variables: list[str],
) -> np.float64:
    logger.trace(f"source energies: {outputs[source_variables].sum(axis=1)}")
    source_energy = integrate.trapezoid(
        -1 * outputs[source_variables].sum(axis=1),
        dx=1 / 4,
    )
    # consumer_energy = integrate.trapezoid(
    #    outputs[consumer_variables].sum(axis=1),
    #    dx=1 / 4,
    # )

    return np.float64(source_energy)


def _compute_gradients(
    outputs: pd.DataFrame,
    source_variables: list[str],
    consumer_variables: list[str],
    controls: list[str],
) -> np.ndarray:
    gradients = []

    for control in controls:
        controlled_source_variables = list(
            map(lambda v: f"{v}|{control}", source_variables)
        )
        source_energy_gradient = integrate.trapezoid(
            -1 * outputs[controlled_source_variables].sum(axis=1),
            dx=1 / 4,
        )
        # consumer_energy_gradient = integrate.trapezoid(
        #    outputs[consumer_variables].sum(axis=1),
        #    dx=1 / 4,
        # )

        gradients.append(source_energy_gradient)

    return np.array(gradients).transpose()


def _compute_min_temperature(
    outputs: pd.DataFrame,
    variables: list[str],
    num_skip_temp_constaints: int,
    min_energy_density: float,
) -> np.ndarray:
    outputs = outputs.iloc[num_skip_temp_constaints:]

    energies = outputs[variables]
    energies -= min_energy_density

    flattened_energies = energies.to_numpy().flatten(order="F")
    return flattened_energies


def _compute_min_temperature_gradients(
    outputs: pd.DataFrame,
    variables: list[str],
    controls: list[str],
    num_skip_temp_constaints: int,
) -> np.ndarray:
    outputs = outputs.iloc[num_skip_temp_constaints:]

    gradients = []

    for control in controls:
        controlled_variables = list(map(lambda v: f"{v}|{control}", variables))
        energy_gradients = outputs[controlled_variables]
        flattened_energy_gradients = energy_gradients.to_numpy().flatten(order="F")
        gradients.append(flattened_energy_gradients)

    gradients_array = np.array(gradients).transpose()

    return gradients_array


def _compute_min_temperature_violation(
    outputs: pd.DataFrame,
    variable: str,
    min_energy_density: float,
) -> np.float64:
    # outputs = outputs.iloc[num_skip_temp_constaints:]

    energies = outputs[variable]
    energies -= min_energy_density
    energies *= -1

    def activation(value):
        return np.maximum(0, value)

    energies = energies.map(lambda e: activation(e))

    return integrate.trapezoid(energies, outputs.index)


def _compute_energy_fed_in(
    outputs: pd.DataFrame,
    source_variables: list[str],
) -> np.float64:
    logger.trace(f"source energies: {outputs[source_variables].sum(axis=1)}")
    source_energy = integrate.trapezoid(
        -1 * outputs[source_variables].sum(axis=1),
        dx=1 / 4,
    )

    return source_energy


def _compute_energy_delivered(
    outputs: pd.DataFrame,
    consumer_power_variables: list[str],
) -> np.float64:
    logger.trace(f"consumer energies: {outputs[consumer_power_variables].sum(axis=1)}")
    source_energy = integrate.trapezoid(
        outputs[consumer_power_variables].sum(axis=1),
        dx=1 / 4,
    )

    return source_energy


def _compute_pump_energy(
    outputs: pd.DataFrame,
    source_names: list[str],
) -> np.float64:
    energy = np.float64(0)

    for name in source_names:
        _energy = integrate.trapezoid(
            outputs[f"{name}.0.q"]
            / 1_000
            * (outputs[f"{name}.1.p"] - outputs[f"{name}.0.p"]),
            dx=1 / 4,
        )
        logger.debug(f"pump energy of {name}: {_energy} (total so far: {energy})")
        energy += _energy

    return energy
