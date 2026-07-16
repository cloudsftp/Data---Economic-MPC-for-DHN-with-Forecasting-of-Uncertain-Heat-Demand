import os
from pathlib import Path
from typing import Callable
import shutil
from collections import OrderedDict

from loguru import logger
import numpy as np
import pandas as pd

# import pydistrictheating_core as dh  # type: ignore

from mpc.time import DAYS, HOURS, MINUTES, SECONDS
from mpc.simulator.parsing import (
    ResultEvaluation,
    evaluate_result_file,
    count_consumers,
)
from mpc.signals import rotate_signal, extend_signal

DEFAULT_TIME_STEP = 15 * MINUTES
MINIMAL_TIME_STEP = 1 * MINUTES
TIME_UNIT = 1 * SECONDS

SIGNAL_LENGTH = 29

CACHE_SIZE = 256


class Simulator:
    num_consumers: int
    num_time_steps: int
    simulation_duration: int

    fixed_control: list[float]

    _simulation_results_path: Path
    _simulation_states_path: Path

    _net: object  # dh.HeatNet

    _iteration: int

    _last_result: pd.DataFrame

    _computed_results: OrderedDict[
        tuple[float],
        ResultEvaluation,
    ]  # maps from hashes of parameters to results

    _num_skip_temp_constraints: int
    _min_temperature: float
    _lowest_temperature_consumer: str = "C5"

    def __init__(
        self,
        network_directory: Path,
        previous_control: list[float],
        previous_demands: dict[str, list[float]],
        simulation_duration: int,
        num_skip_temp_constraints: int,
        min_temperature: float,
    ) -> None:
        self._num_skip_temp_constraints = num_skip_temp_constraints
        self._min_temperature = min_temperature
        self.simulation_duration = simulation_duration

        network_directory = network_directory.absolute()
        logger.trace(f"network directory: {network_directory}")

        self._configuration_file_path = network_directory.joinpath("main.json")
        logger.trace(
            f"setting configuration file path to {self._configuration_file_path}"
        )
        assert self._configuration_file_path.exists()

        self._net = object()  # dh.HeatNet("net")
        logger.trace(f"loading configuration: {str(self._configuration_file_path)}")
        self._net._load(str(self._configuration_file_path))

        self._simulation_results_path = network_directory.joinpath("results")
        os.makedirs(self._simulation_results_path, exist_ok=True)

        self._simulation_states_path = network_directory.joinpath("states")
        os.makedirs(self._simulation_states_path, exist_ok=True)

        self._iteration = 0
        self._net._initParameters()
        self.set_controls(previous_control, skip_fixed=True)
        for key in previous_demands:
            previous_demands[key] = previous_demands[key][-25:]
        self.set_demands(previous_demands)
        self._set_simulation_time(0, 1 * DAYS)
        self._set_ramp(4 * HOURS)
        self._net._simulate()
        self._save_states()

        self._set_simulation_time(0, self.simulation_duration)
        self._set_ramp(0)

        self._computed_results = OrderedDict()

        self.num_time_steps = int(self.simulation_duration / DEFAULT_TIME_STEP) + 1
        result_file_path = self._iteration_simulation_results_path().joinpath("result")
        self._export_outputs(result_file_path)
        self.num_consumers = count_consumers(result_file_path)

        self.fixed_control = previous_control[-1:]

        self._last_result = pd.DataFrame()

    def advance_step(self, control: list[float]):
        shutil.rmtree(
            self._iteration_simulation_results_path(),
            ignore_errors=True,
        )

        self.set_controls(control)
        self.set_simulation_time_for_simulation(1 * HOURS)
        self._simulate()

        self._iteration += 1
        self._save_states()

        self.set_simulation_time_for_simulation()
        self._computed_results = OrderedDict()

        self.fixed_control = self.fixed_control[1:] + [control[0]]

    def set_simulation_time_for_simulation(self, duration: int | None = None) -> None:
        if not duration:
            duration = self.simulation_duration

        self._set_simulation_time(self._iteration * HOURS, duration)

    def _set_simulation_time(self, start: float, duration: float):
        self._net.setTimes(
            start,
            DEFAULT_TIME_STEP,
            start + duration,
            TIME_UNIT,
            MINIMAL_TIME_STEP,
        )

    def _simulate(self):
        logger.trace(
            f"attempting simulation with initial states file '{self._initial_states_file_path()}'"
        )
        self._net._simulate(initStateFilename=str(self._initial_states_file_path()))

    def _initial_states_file_path(self) -> Path:
        return self._simulation_states_path.joinpath(f"initial_{self._iteration:03d}")

    def _iteration_simulation_results_path(self) -> Path:
        return self._simulation_results_path.joinpath(f"{self._iteration:03}")

    def _save_states(self):
        self._net.saveStates(str(self._initial_states_file_path()))

    def _set_ramp(self, duration: float):
        self._net.setRamp(duration)

    def set_controls(self, parameters: list[float], skip_fixed: bool = False):
        control = parameters
        if not skip_fixed:
            control = self.fixed_control + control
        control = extend_signal(control, SIGNAL_LENGTH)
        control = rotate_signal(control, self._iteration)
        self._net._updateParameters(control)

    def set_demands(self, demands: dict[str, list[float]]):
        for name in demands:
            demand = demands[name]
            demand = extend_signal(demand, SIGNAL_LENGTH)
            demand = rotate_signal(demand, self._iteration)
            self._net.update_signal_values(f"{name}_demand", demand)

    def _export_outputs(self, result_file_path: Path):
        self._net.exportOutputs(str(result_file_path), True)

    def _evaluate(self, parameters: np.ndarray) -> ResultEvaluation:
        logger.debug(f"evaluating simulation for parameters {parameters}")
        parameters_tuple = tuple(parameters)

        if parameters_tuple in self._computed_results:
            result = self._computed_results[parameters_tuple]
            logger.trace(f"found a result for this set of parameters: {result}")
            return result

        logger.trace("did not find a result for this set of parameters, simulating now")
        self.set_controls(parameters.tolist())
        self._simulate()

        result_file_path = self._iteration_simulation_results_path().joinpath(
            f"result_{hash(parameters_tuple)}"
        )
        logger.trace(f"exporting simulation results to '{result_file_path}'")
        self._export_outputs(result_file_path)

        result = evaluate_result_file(
            result_file_path,
            self.num_time_steps,
            len(self.fixed_control),
            len(parameters),
            self._iteration,
            self._num_skip_temp_constraints,
            self._min_temperature,
            self._lowest_temperature_consumer,
        )

        if len(self._computed_results) > CACHE_SIZE:
            self._computed_results.popitem(last=False)

        self._computed_results[parameters_tuple] = result
        self._last_result = result.result
        logger.trace(f"result: {result}")
        return result

    def objective(self) -> Callable[[np.ndarray], np.float64]:
        def o(parameters: np.ndarray) -> np.float64:
            result = self._evaluate(parameters)
            return result.objective

        return o

    def jacobian(self) -> Callable[[np.ndarray], np.ndarray]:
        def g(parameters: np.ndarray) -> np.ndarray:
            result = self._evaluate(parameters)
            return result.gradients

        return g

    def min_temperature(self) -> Callable[[np.ndarray], np.ndarray]:
        def t(parameters: np.ndarray) -> np.ndarray:
            result = self._evaluate(parameters)
            return result.min_temperature

        return t

    def min_temperature_gradients(self) -> Callable[[np.ndarray], np.ndarray]:
        def g(parameters: np.ndarray) -> np.ndarray:
            result = self._evaluate(parameters)
            return result.min_temperature_gradients

        return g

    def min_temperature_violation(self) -> Callable[[np.ndarray], np.float64]:
        def v(parameters: np.ndarray) -> np.float64:
            result = self._evaluate(parameters)
            return result.min_temperature_violation

        return v
