from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from loguru import logger
import numpy as np

from mpc.demand import DemandProviderI
from mpc.optimizer import Optimizer, OptimizerUpdate
from mpc import visualize
from mpc.optimizer.objective import ObjectiveFunction


@dataclass
class MPCResult:
    optimal_controls: list[list[float]]
    optimal_control: list[float]

    times: list[float]
    iterations: list[int]

    @staticmethod
    def from_optimal_controls(
        optimal_controls: list[list[float]],
        times: list[float],
        iterations: list[int],
    ) -> MPCResult:
        optimal_control = [optimal_controls[0][0]]
        for i in range(len(optimal_controls)):
            optimal_control.append(optimal_controls[i][1])

        return MPCResult(
            optimal_controls,
            optimal_control,
            times,
            iterations,
        )


class MPC:
    optimizer: Optimizer
    demand_provider: DemandProviderI

    mpc_steps: int

    _objective_values: list[float]

    objective_values: list[float]
    objective_short_values: list[float]
    ells: list[float]

    def __init__(
        self,
        network_directory: Path,
        objective_function: ObjectiveFunction,
        demand_provider: DemandProviderI,
        previous_control_temperatures: list[float],
        initial_control_temperature: float,
        num_controls: int,
        horizon_hours: int,
        num_steps: int,
        min_temperature: float,
    ) -> None:
        previous_control = previous_control_temperatures
        initial_control = [initial_control_temperature] * num_controls

        self.demand_provider = demand_provider

        self.demand_provider.assert_enough_data_available(num_steps)
        previous_demands = self.demand_provider.get_previous_demands()
        predicted_demands = self.demand_provider.get_predicted_future_demands()

        self._objective_values = []

        self.objective_values = []
        self.objective_short_values = []
        self.ells = []

        def visualize_progress(update: OptimizerUpdate):
            visualize.plot_control(update.current_control, self.optimizer._demands)

            self._objective_values.append(update.current_objective_value)
            visualize.plot_objective_values(self._objective_values)

            visualize.plot_simulation_result(update.simulation_result)

        self.optimizer = Optimizer(
            network_directory,
            objective_function,
            previous_control,
            previous_demands,
            initial_control,
            predicted_demands,
            horizon_hours,
            min_temperature,
            report_intermediate_result=visualize_progress,
        )

        self.mpc_steps = num_steps

    def visualize_iteration_progress(self, update: OptimizerUpdate):
        visualize.plot_control(
            update.current_control,
            self.optimizer._demands,
            Path(f"results/control_{update.iteration}.png"),
        )

        visualize.plot_simulation_result(
            update.simulation_result,
            Path(f"results/simulation_temperatures_{update.iteration}.png"),
        )

    def run(self) -> MPCResult:
        logger.info("Starting mpc loop")

        optimal_controls = []
        times = []
        iterations = []

        for step in range(self.mpc_steps):
            logger.info(f"\n===== MPC step {step} =====")
            optimization_result = self.optimizer.optimize()
            logger.info(
                f"success: {optimization_result.optimized}, control: {optimization_result.control}"
            )

            optimal_controls.append(optimization_result.control)
            times.append(optimization_result.time)
            iterations.append(optimization_result.iterations)

            actual_demands = self.demand_provider.get_actual_future_demands()
            self.demand_provider.advance_step()
            predicted_demands = self.demand_provider.get_predicted_future_demands()

            self.visualize_iteration_progress(
                self.optimizer.get_optimizer_update(
                    np.array(optimization_result.control),
                ),
            )

            self.optimizer.advance_step(actual_demands, predicted_demands)

            # just for visualization
            self._objective_values = []

        return MPCResult.from_optimal_controls(
            optimal_controls,
            times,
            iterations,
        )
