from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from scipy import stats
from serde.json import to_json, from_json
import pandas as pd

from mpc import visualize
from mpc.config import (
    DEFAULT_NETWORK_DIRECTORY,
    FutureDataMode,
    get_demand_provider,
)
from mpc.water import MIN_TEMPERATURE, temperature
from mpc.optimizer import Optimizer
from mpc.optimizer.objective import J1, J2
from mpc.mpc import MPC
from mpc.simulator import Simulator
from mpc.simulator.parsing import (
    _read_output_csv,
)

CONSTANT_TEMPERATURE = 74.0
INITIAL_TEMPERATURE = 75.0

FULL_DAY_HORIZON_HOURS = 24
FULL_DAY_HORIZON_HOURS_EXTENDED = 28
MPC_HORIZON_HOURS = 8
MPC_STEPS = 24

C5_FEED_ENERGY_DENSITY_COLUMN = "C5.0.e"

HOURS = 60 * 60


def run_experiments(
    initial_base_hour: int,
    start_day_offset: int,
    num_days: int,
    output_directory: Path,
) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)

    for day in range(start_day_offset, num_days):
        experiment_output_file = (
            output_directory / f"day-optimization-results-{day:02d}.json"
        )
        day_optimization_results = run_experiment(
            day,
            initial_base_hour + day * 24,
        )

        with open(experiment_output_file, "w") as f:
            f.write(to_json(day_optimization_results, indent=2))


@dataclass
class Evaluation:
    control: list[float]
    objective: float
    modified_objective: float
    maximum: float
    area: float


@dataclass
class DayEvaluations:
    day: int
    base_hour: int
    previous_control_no_buffer: list[float]
    # evaluation_open_loop_real: Evaluation
    evaluation_open_loop_forecasted: Evaluation
    evaluation_mpc_no_buffer: Evaluation
    # evaluation_mpc_with_buffer: Evaluation


@dataclass
class DayOptimizationResults:
    day: int
    base_hour: int
    previous_control_no_buffer: list[float]
    # control_open_loop_real: list[float]
    control_open_loop_forecasted: list[float]
    control_mpc_no_buffer: list[float]
    # control_mpc_with_buffer: list[float]


def run_experiment(
    day: int,
    base_hour: int,
) -> DayOptimizationResults:
    previous_control_no_buffer = compute_previous_control(
        base_hour,
        MIN_TEMPERATURE,
    )

    control_open_loop_forecasted = run_open_loop(
        FutureDataMode.Forecasted,
        base_hour,
        previous_control_no_buffer,
        MIN_TEMPERATURE,
    )
    control_mpc_no_buffer = run_mpc(
        base_hour,
        previous_control_no_buffer,
        MIN_TEMPERATURE,
    )

    return DayOptimizationResults(
        day,
        base_hour,
        previous_control_no_buffer,
        control_open_loop_forecasted,
        control_mpc_no_buffer,
    )


def compute_previous_control(
    base_hour: int,
    min_temperature: float,
) -> list[float]:
    horizon_hours = FULL_DAY_HORIZON_HOURS
    previous_control = [CONSTANT_TEMPERATURE] * (horizon_hours + 1)

    control = simple_optimization(
        FutureDataMode.Actual,
        base_hour - 12,
        horizon_hours,
        previous_control,
        min_temperature,
    )
    return [CONSTANT_TEMPERATURE] * 12 + control[:13]


def run_open_loop(
    mode: FutureDataMode,
    base_hour: int,
    previous_control: list[float],
    min_temperature: float,
) -> list[float]:
    return simple_optimization(
        mode,
        base_hour,
        FULL_DAY_HORIZON_HOURS_EXTENDED,
        previous_control,
        min_temperature,
    )[1 : FULL_DAY_HORIZON_HOURS + 1]


def simple_optimization(
    mode: FutureDataMode,
    base_hour: int,
    horizon_hours: int,
    previous_control: list[float],
    min_temperature: float,
) -> list[float]:
    num_controls = horizon_hours
    initial_control = [INITIAL_TEMPERATURE] * num_controls

    demand_provider = get_demand_provider(
        mode,
        base_hour,
        horizon_hours,
    )
    previous_demands = demand_provider.get_previous_demands()
    demands = demand_provider.get_predicted_future_demands()

    opt = Optimizer(
        DEFAULT_NETWORK_DIRECTORY,
        J1(),
        previous_control,
        previous_demands,
        initial_control,
        demands,
        horizon_hours,
        min_temperature,
    )
    optimization_result = opt.optimize()

    return optimization_result.control


def run_mpc(
    base_hour: int,
    previous_control: list[float],
    min_temperature: float,
) -> list[float]:
    horizon_hours = MPC_HORIZON_HOURS
    num_controls = horizon_hours

    demand_provider = get_demand_provider(
        FutureDataMode.Forecasted,
        base_hour,
        horizon_hours,
    )

    num_steps = MPC_STEPS

    mpc = MPC(
        DEFAULT_NETWORK_DIRECTORY,
        J1(),
        demand_provider,
        previous_control,
        INITIAL_TEMPERATURE,
        num_controls,
        horizon_hours,
        num_steps,
        min_temperature,
    )
    mpc_results = mpc.run()

    return mpc_results.optimal_control[1 : FULL_DAY_HORIZON_HOURS + 1]


def simulate_control(
    base_hour: int,
    previous_control: list[float],
    control: list[float],
) -> tuple[Simulator, pd.DataFrame]:
    network_directory = DEFAULT_NETWORK_DIRECTORY

    horizon_hours = FULL_DAY_HORIZON_HOURS
    demand_provider = get_demand_provider(
        FutureDataMode.Actual,
        base_hour,
        horizon_hours,
    )

    previous_demands = demand_provider.get_previous_demands()
    future_demands = demand_provider.get_predicted_future_demands()

    simulator = Simulator(
        network_directory,
        previous_control,
        previous_demands,
        horizon_hours * HOURS,
        0,
        MIN_TEMPERATURE,
    )
    simulator.set_demands(future_demands)

    simulator.set_controls(control)
    simulator._simulate()
    result_file = network_directory / "evaluate/tmp.csv"
    result_file.parent.mkdir(parents=True, exist_ok=True)
    simulator._export_outputs(Path("evaluate/tmp.csv"))
    results_df = _read_output_csv(result_file)

    return simulator, results_df


def evaluate_control(
    base_hour: int,
    previous_control: list[float],
    control: list[float],
) -> Evaluation:
    simulator, results_df = simulate_control(
        base_hour,
        previous_control,
        control,
    )

    # compute objective

    objective = J1().build_objective(simulator)(np.array(control[1:]))

    # compute violation metrics

    time_hours = results_df.index.to_numpy(dtype=float) / HOURS
    feed_temperature_C5 = temperature(results_df[C5_FEED_ENERGY_DENSITY_COLUMN])
    violation = (MIN_TEMPERATURE - feed_temperature_C5).map(lambda v: max(0, v))

    area = np.trapezoid(violation, x=time_hours)
    maximum = np.float64(violation.max())

    # compute modified objective

    modified_objective = J2().build_objective(simulator)(np.array(control[1:]))

    return Evaluation(
        control,
        objective,
        modified_objective,
        maximum,
        float(area),
    )


def evaluate_day_optimization_results(
    day_optimization_results: DayOptimizationResults,
) -> DayEvaluations:
    base_hour = day_optimization_results.base_hour
    previous_control_no_buffer = day_optimization_results.previous_control_no_buffer

    evaluation_open_loop_forecasted = evaluate_control(
        base_hour,
        previous_control_no_buffer,
        day_optimization_results.control_open_loop_forecasted,
    )
    evaluation_mpc_no_buffer = evaluate_control(
        base_hour,
        previous_control_no_buffer,
        day_optimization_results.control_mpc_no_buffer,
    )

    return DayEvaluations(
        day_optimization_results.day,
        base_hour,
        previous_control_no_buffer,
        evaluation_open_loop_forecasted,
        evaluation_mpc_no_buffer,
    )


@dataclass
class Distribution:
    n: int
    mean: float
    std: float
    minimum: float
    q1: float
    median: float
    q3: float
    maximum: float

    @staticmethod
    def from_values(values: list[float]) -> Distribution:
        array = np.asarray(values, dtype=float)
        described = stats.describe(array)
        return Distribution(
            n=int(described.nobs),
            mean=float(described.mean),
            std=float(np.std(array, ddof=1)) if array.size > 1 else 0.0,
            minimum=float(described.minmax[0]),
            q1=float(np.percentile(array, 25)),
            median=float(np.median(array)),
            q3=float(np.percentile(array, 75)),
            maximum=float(described.minmax[1]),
        )


@dataclass
class StrategyDistributions:
    objective: Distribution
    modified_objective: Distribution
    maximum: Distribution
    area: Distribution


@dataclass
class StrategyImprovements:
    days: list[int]
    objective: list[float]
    modified_objective: list[float]
    maximum: list[float]
    area: list[float]
    distributions: StrategyDistributions


@dataclass
class MultiDayDistributions:
    open_loop_forecasted: StrategyDistributions
    mpc_no_buffer: StrategyDistributions
    mpc_no_buffer_improvement: StrategyImprovements


def evaluate_optimization_results(
    data_directory: Path,
    num_days: Optional[int] = None,
) -> None:
    files = sorted(data_directory.glob("day-optimization-results-*.json"))

    if num_days:
        files = files[:num_days]
    for filepath in files:
        with filepath.open() as f:
            day_optimization_results = from_json(DayOptimizationResults, f.read())

        day_evaluations = evaluate_day_optimization_results(day_optimization_results)
        evaluations_filename = f"day-evaluations-{day_evaluations.day:02d}.json"

        with (data_directory / evaluations_filename).open("w") as f:
            f.write(to_json(day_evaluations, indent=2))


STRATEGIES = (
    "open_loop_forecasted",
    "mpc_no_buffer",
)


def compute_evaluation_distributions(
    data_directory: Path,
    num_days: Optional[int] = None,
) -> MultiDayDistributions:
    files = sorted(data_directory.glob("day-evaluations-*.json"))

    if num_days:
        files = files[:num_days]

    days: list[int] = []
    by_strategy: dict[str, list[Evaluation]] = {name: [] for name in STRATEGIES}

    for filepath in files:
        with open(filepath) as f:
            day_evaluations = from_json(DayEvaluations, f.read())

        days.append(day_evaluations.day)
        by_strategy["open_loop_forecasted"].append(
            day_evaluations.evaluation_open_loop_forecasted
        )
        by_strategy["mpc_no_buffer"].append(day_evaluations.evaluation_mpc_no_buffer)

    def _strategy_distributions(evaluations: list[Evaluation]) -> StrategyDistributions:
        return StrategyDistributions(
            objective=Distribution.from_values([e.objective for e in evaluations]),
            modified_objective=Distribution.from_values(
                [e.modified_objective for e in evaluations]
            ),
            maximum=Distribution.from_values([e.maximum for e in evaluations]),
            area=Distribution.from_values([e.area for e in evaluations]),
        )

    def _improvements(
        baseline: list[Evaluation],
        improved: list[Evaluation],
    ) -> StrategyImprovements:
        objective = [
            b.objective - i.objective for b, i in zip(baseline, improved, strict=True)
        ]
        modified_objective = [
            b.modified_objective - i.modified_objective
            for b, i in zip(baseline, improved, strict=True)
        ]
        maximum = [
            b.maximum - i.maximum for b, i in zip(baseline, improved, strict=True)
        ]
        area = [b.area - i.area for b, i in zip(baseline, improved, strict=True)]

        return StrategyImprovements(
            days,
            objective,
            modified_objective,
            maximum,
            area=area,
            distributions=StrategyDistributions(
                objective=Distribution.from_values(objective),
                modified_objective=Distribution.from_values(modified_objective),
                maximum=Distribution.from_values(maximum),
                area=Distribution.from_values(area),
            ),
        )

    return MultiDayDistributions(
        open_loop_forecasted=_strategy_distributions(
            by_strategy["open_loop_forecasted"]
        ),
        mpc_no_buffer=_strategy_distributions(by_strategy["mpc_no_buffer"]),
        mpc_no_buffer_improvement=_improvements(
            by_strategy["open_loop_forecasted"],
            by_strategy["mpc_no_buffer"],
        ),
    )


def plot_evaluation_distributions(
    data_directory: Path,
    output_file: Path,
    num_days: Optional[int] = None,
) -> None:
    distributions = compute_evaluation_distributions(data_directory, num_days)
    visualize.plot_evaluation_distributions(distributions, output_file)


def simulate_and_plot_results(
    data_directory: Path,
    output_directory: Path,
    num_days: Optional[int] = None,
) -> None:
    files = sorted(data_directory.glob("day-optimization-results-*.json"))

    output_directory.mkdir(parents=True, exist_ok=True)

    if num_days:
        files = files[:num_days]
    for filepath in files:
        with open(filepath) as f:
            day_optimization_results = from_json(DayOptimizationResults, f.read())

        _, open_loop_result_df = simulate_control(
            day_optimization_results.base_hour,
            day_optimization_results.previous_control_no_buffer,
            day_optimization_results.control_open_loop_forecasted,
        )
        visualize.plot_simulation_result(
            open_loop_result_df,
            output_directory
            / f"open-loop-forecasted-day{day_optimization_results.day:02d}.png",
            MIN_TEMPERATURE,
        )

        _, mpc_no_buffer_result_df = simulate_control(
            day_optimization_results.base_hour,
            day_optimization_results.previous_control_no_buffer,
            day_optimization_results.control_mpc_no_buffer,
        )
        visualize.plot_simulation_result(
            mpc_no_buffer_result_df,
            output_directory
            / f"mpc-no-buffer-day{day_optimization_results.day:02d}.png",
            MIN_TEMPERATURE,
        )
