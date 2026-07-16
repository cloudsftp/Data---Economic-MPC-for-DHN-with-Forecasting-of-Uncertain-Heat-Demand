from pathlib import Path
from typing import Callable, Optional
import time
from dataclasses import dataclass

from loguru import logger
import pandas as pd
import numpy as np
from scipy import optimize
from scipy.optimize import (
    NonlinearConstraint,
    OptimizeResult,
)

from mpc.simulator import Simulator
from mpc.time import HOURS, MINUTES
from mpc.control import changes_in_temperature_constraint, boundaries
from mpc.optimizer.objective import ObjectiveFunction


@dataclass
class OptimizerUpdate:
    iteration: int
    current_objective_value: float
    current_control: list[float]
    current_demands: dict[str, list[float]]
    simulation_result: pd.DataFrame


@dataclass
class OptimizationResult:
    control: list[float]
    optimized: bool

    time: float
    iterations: int


class Optimizer:
    _objective_function: ObjectiveFunction

    _simulator: Simulator
    _num_consumer_temperatures: int
    _num_skip_temp_constraints: int
    _num_control: int

    _optimized: bool
    _optimal_control: list[float]
    _initial_control: list[float]

    _demands: dict[str, list[float]]

    _report_intermediate_result: Optional[Callable[[OptimizerUpdate], None]]
    _current_objective_value: float

    _ftol: float

    def __init__(
        self,
        network_directory: Path,
        objective_function: ObjectiveFunction,
        previous_control: list[float],
        previous_demands: dict[str, list[float]],
        initial_control: list[float],
        initial_demands: dict[str, list[float]],
        simulation_duration_hours: int,
        min_temperature: float,
        report_intermediate_result: Optional[Callable[[OptimizerUpdate], None]] = None,
        hours_ignore_temp_constraints: float = 3.5,
    ):
        self._objective_function = objective_function

        self._ftol = simulation_duration_hours * 10

        self._num_skip_temp_constraints = int(
            hours_ignore_temp_constraints * HOURS / (15 * MINUTES)
        )
        self._simulator = Simulator(
            network_directory,
            previous_control,
            previous_demands,
            simulation_duration_hours * HOURS,
            self._num_skip_temp_constraints,
            min_temperature,
        )
        self._num_consumer_temperatures = self._simulator.num_consumers * (
            self._simulator.num_time_steps - self._simulator._num_skip_temp_constraints
        )

        self._num_control = len(initial_control)
        self._initial_control = initial_control
        self._optimal_control = self._initial_control

        self._simulator.set_controls(initial_control)
        self._simulator.set_demands(initial_demands)

        self._demands = initial_demands

        self._report_intermediate_result = report_intermediate_result

    def optimize(self) -> OptimizationResult:
        objective = self._objective_function.build_objective(self._simulator)
        jacobian = self._objective_function.build_jacobian(self._simulator)

        def wrapped_objective(parameters: np.ndarray) -> float:
            value = objective(parameters)

            self._current_objective_value = float(value)

            return value

        changes_in_temperature_in_range = changes_in_temperature_constraint(
            self._num_control,
        )

        temperature_above_70 = NonlinearConstraint(
            self._simulator.min_temperature(),
            lb=np.zeros(self._num_consumer_temperatures),
            ub=np.inf * np.ones(self._num_consumer_temperatures),
            jac=self._simulator.min_temperature_gradients(),
        )

        constraints = (
            changes_in_temperature_in_range,
            temperature_above_70,
        )

        bounds = boundaries(
            self._num_control,
            self._simulator.fixed_control[-1],
        )

        initial_control = self._initial_control
        previous_objective_value: Optional[float] = None
        small_objective_changes = 0

        def callback(
            intermediate_result: OptimizeResult,
        ):
            nonlocal previous_objective_value
            nonlocal small_objective_changes

            control = intermediate_result.x
            objective_value = float(intermediate_result.fun)

            self._current_objective_value = objective_value
            self.update_result(control)

            previous_objective_value = objective_value

        start_time = time.time()
        result = optimize.minimize(
            wrapped_objective,
            initial_control,
            jac=jacobian,
            bounds=bounds,
            constraints=constraints,
            method="SLSQP",
            callback=callback,
            options={
                "ftol": self._ftol,
                "maxiter": 10,
            },
        )

        duration = time.time() - start_time
        iterations = result.nit

        logger.info(f"minimize returned: {result}")
        logger.info(
            f"optimization terminated after {duration} s with {iterations} iterations"
        )

        callback(result)

        self._optimal_control = result.x.tolist()

        self._optimized = True

        return OptimizationResult(
            self.combine_with_fixed_control(self._optimal_control),
            self._optimized,
            duration,
            iterations,
        )

    def advance_step(
        self,
        actual_demands: dict[str, list[float]],
        predicted_demands: dict[str, list[float]],
    ):
        self._optimized = False
        self._simulator.set_demands(actual_demands)
        self._simulator.advance_step(self._optimal_control)

        demands = dict()
        for name in predicted_demands:
            demands[name] = [actual_demands[name][1]] + predicted_demands[name][1:]

        self._simulator.set_demands(demands)
        self._demands = demands

    def update_result(self, control: np.ndarray):
        if self._report_intermediate_result:
            self._report_intermediate_result(
                self.get_optimizer_update(control),
            )

    def combine_with_fixed_control(self, optimal_control: list[float]) -> list[float]:
        return self._simulator.fixed_control + optimal_control

    def get_optimizer_update(self, control: np.ndarray):
        return OptimizerUpdate(
            self._simulator._iteration,
            self._current_objective_value,
            self.combine_with_fixed_control(control.tolist()),
            self._demands,
            self._simulator._last_result,
        )
