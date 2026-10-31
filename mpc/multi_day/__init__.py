from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Optional, Sequence

from loguru import logger
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
from mpc.curve import Curve
from mpc.water import MIN_TEMPERATURE, MIN_TEMPERATURE_W_BUFFER, temperature
from mpc.optimizer import Optimizer
from mpc.optimizer.objective import J1, J2
from mpc.mpc import MPC, MPCResult
from mpc.simulator import Simulator
from mpc.simulator.parsing import (
    ResultEvaluation,
    _read_output_csv,
    evaluate_results,
)

CONSTANT_TEMPERATURE = 76.0
INITIAL_TEMPERATURE = 76.0

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
    skip_opt: bool,
) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)

    for day in range(start_day_offset, num_days):
        day_optimization_results = run_experiment(
            day,
            initial_base_hour + day * 24,
            skip_opt,
        )

        experiment_output_file = (
            output_directory / f"day-optimization-results-{day:02d}.json"
        )
        if skip_opt:
            tmp_optimization_results = day_optimization_results

            with open(experiment_output_file, "r") as f:
                day_optimization_results = from_json(DayOptimizationResults, f.read())

            day_optimization_results.control_curve = (
                tmp_optimization_results.control_curve
            )
            day_optimization_results.previous_control_curve = (
                tmp_optimization_results.previous_control_curve
            )

        with open(experiment_output_file, "w") as f:
            f.write(to_json(day_optimization_results, indent=2))


@dataclass
class Evaluation:
    control: list[float]
    objective: float
    modified_objective: float
    heat_fed_in: float
    heat_delivered: float
    heat_loss: float
    pump_energy: float
    maximum: float
    area: float


@dataclass
class DayEvaluations:
    day: int
    base_hour: int
    previous_control_no_buffer: list[float]
    evaluation_open_loop_forecasted: Evaluation
    evaluation_open_loop_forecasted_w_buffer: Evaluation
    evaluation_mpc_no_buffer: Evaluation
    evaluation_curve: Evaluation


@dataclass
class DayOptimizationResults:
    day: int
    base_hour: int
    previous_control_no_buffer: list[float] = field(default_factory=list)
    previous_control_w_buffer: list[float] = field(default_factory=list)
    previous_control_curve: list[float] = field(default_factory=list)
    control_open_loop_forecasted: list[float] = field(default_factory=list)
    control_open_loop_forecasted_w_buffer: list[float] = field(default_factory=list)
    control_mpc_no_buffer: list[float] = field(default_factory=list)
    times_mpc: list[float] = field(default_factory=list)
    iterations_mpc: list[int] = field(default_factory=list)
    successes_mpc: list[bool] = field(default_factory=list)
    stati_mpc: list[int] = field(default_factory=list)
    messages_mpc: list[str] = field(default_factory=list)
    control_curve: list[float] = field(default_factory=list)


def run_experiment(
    day: int,
    base_hour: int,
    skip_opt: bool,
) -> DayOptimizationResults:
    if not skip_opt:
        previous_control_no_buffer = compute_previous_control(
            base_hour,
            MIN_TEMPERATURE,
        )
        previous_control_w_buffer = compute_previous_control(
            base_hour,
            MIN_TEMPERATURE_W_BUFFER,
        )

        control_open_loop_forecasted = run_open_loop(
            FutureDataMode.Forecasted,
            base_hour,
            previous_control_no_buffer,
            MIN_TEMPERATURE,
        )
        control_open_loop_forecasted_w_buffer = run_open_loop(
            FutureDataMode.Forecasted,
            base_hour,
            previous_control_w_buffer,
            MIN_TEMPERATURE_W_BUFFER,
        )

        mpc_result = run_mpc(
            base_hour,
            previous_control_no_buffer,
            MIN_TEMPERATURE,
        )
        control_mpc_no_buffer = mpc_result.optimal_control[
            1 : FULL_DAY_HORIZON_HOURS + 1
        ]

    else:
        previous_control_no_buffer = []
        previous_control_w_buffer = []

        control_open_loop_forecasted = []
        control_open_loop_forecasted_w_buffer = []

        mpc_result = MPCResult([], [], [], [], [], [], [])
        control_mpc_no_buffer = []

    control_curve = Curve(base_hour).control()
    previous_control_curve = Curve(base_hour - 24).control()

    return DayOptimizationResults(
        day,
        base_hour,
        previous_control_no_buffer,
        previous_control_w_buffer,
        previous_control_curve,
        control_open_loop_forecasted,
        control_open_loop_forecasted_w_buffer,
        control_mpc_no_buffer,
        mpc_result.times,
        mpc_result.iterations,
        mpc_result.successes,
        mpc_result.stati,
        mpc_result.messages,
        control_curve,
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
) -> MPCResult:
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
    return mpc.run()


def simulate_control(
    base_hour: int,
    previous_control: list[float],
    min_temperature: float,
    control: list[float],
) -> tuple[Simulator, pd.DataFrame, ResultEvaluation]:
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
        0,
        min_temperature,
    )
    simulator.set_demands(future_demands)

    simulator.set_controls(control)
    simulator._simulate()
    result_file = network_directory / "evaluate/tmp.csv"
    result_file.parent.mkdir(parents=True, exist_ok=True)
    simulator._export_outputs(Path("evaluate/tmp.csv"))
    results_df = _read_output_csv(result_file)
    evaluation = evaluate_results(
        results_df,
        simulator.num_time_steps,
        len(simulator.fixed_control),
        24,
        0,
        0,
        min_temperature,
        "C5",
    )

    return simulator, results_df, evaluation


def evaluate_control(
    base_hour: int,
    previous_control: list[float],
    min_temperature: float,
    control: list[float],
) -> Evaluation:
    simulator, results_df, evaluation = simulate_control(
        base_hour,
        previous_control,
        min_temperature,
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
        evaluation.heat_fed_in,
        evaluation.heat_delivered,
        evaluation.heat_loss,
        evaluation.pump_energy,
        maximum,
        float(area),
    )


def evaluate_day_optimization_results(
    day_optimization_results: DayOptimizationResults,
) -> DayEvaluations:
    base_hour = day_optimization_results.base_hour
    previous_control_no_buffer = day_optimization_results.previous_control_no_buffer
    previous_control_w_buffer = day_optimization_results.previous_control_w_buffer
    previous_control_curve = day_optimization_results.previous_control_curve

    evaluation_open_loop_forecasted = evaluate_control(
        base_hour,
        previous_control_no_buffer,
        MIN_TEMPERATURE,
        day_optimization_results.control_open_loop_forecasted,
    )
    evaluation_open_loop_forecasted_w_buffer = evaluate_control(
        base_hour,
        previous_control_w_buffer,
        MIN_TEMPERATURE_W_BUFFER,
        day_optimization_results.control_open_loop_forecasted_w_buffer,
    )
    evaluation_mpc_no_buffer = evaluate_control(
        base_hour,
        previous_control_no_buffer,
        MIN_TEMPERATURE,
        day_optimization_results.control_mpc_no_buffer,
    )
    evaluation_curve = evaluate_control(
        base_hour,
        previous_control_curve,
        MIN_TEMPERATURE,
        day_optimization_results.control_curve,
    )

    return DayEvaluations(
        day_optimization_results.day,
        base_hour,
        previous_control_no_buffer,
        evaluation_open_loop_forecasted,
        evaluation_open_loop_forecasted_w_buffer,
        evaluation_mpc_no_buffer,
        evaluation_curve,
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
    heat_fed_in: Distribution
    heat_delivered: Distribution
    heat_loss: Distribution
    pump_energy: Distribution
    maximum: Distribution
    area: Distribution


@dataclass
class StrategyImprovements:
    days: list[int]
    objective: list[float]
    modified_objective: list[float]
    maximum: list[float]
    pump_energy: list[float]
    area: list[float]
    distributions: StrategyDistributions


@dataclass
class MultiDayDistributions:
    open_loop_forecasted: StrategyDistributions
    open_loop_forecasted_w_buffer: StrategyDistributions
    mpc_no_buffer: StrategyDistributions
    mpc_no_buffer_improvement: StrategyImprovements
    curve: StrategyDistributions


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
    "open_loop_forecasted_w_buffer",
    "mpc_no_buffer",
    "curve",
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
            day_evaluations.evaluation_open_loop_forecasted,
        )
        by_strategy["open_loop_forecasted_w_buffer"].append(
            day_evaluations.evaluation_open_loop_forecasted_w_buffer,
        )
        by_strategy["mpc_no_buffer"].append(day_evaluations.evaluation_mpc_no_buffer)
        by_strategy["curve"].append(day_evaluations.evaluation_curve)

    def _strategy_distributions(evaluations: list[Evaluation]) -> StrategyDistributions:
        return StrategyDistributions(
            objective=Distribution.from_values([e.objective for e in evaluations]),
            heat_fed_in=Distribution.from_values([e.heat_fed_in for e in evaluations]),
            heat_delivered=Distribution.from_values(
                [e.heat_delivered for e in evaluations]
            ),
            heat_loss=Distribution.from_values([e.heat_loss for e in evaluations]),
            pump_energy=Distribution.from_values([e.pump_energy for e in evaluations]),
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
        heat_fed_in = [
            b.heat_fed_in - i.heat_fed_in
            for b, i in zip(baseline, improved, strict=True)
        ]
        heat_delivered = [
            b.heat_delivered - i.heat_delivered
            for b, i in zip(baseline, improved, strict=True)
        ]
        heat_loss = [
            b.heat_loss - i.heat_loss for b, i in zip(baseline, improved, strict=True)
        ]
        pump_energy = [
            b.pump_energy - i.pump_energy
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
            pump_energy,
            maximum,
            area,
            distributions=StrategyDistributions(
                objective=Distribution.from_values(objective),
                modified_objective=Distribution.from_values(modified_objective),
                heat_fed_in=Distribution.from_values(heat_fed_in),
                heat_delivered=Distribution.from_values(heat_delivered),
                heat_loss=Distribution.from_values(heat_loss),
                pump_energy=Distribution.from_values(pump_energy),
                maximum=Distribution.from_values(maximum),
                area=Distribution.from_values(area),
            ),
        )

    return MultiDayDistributions(
        open_loop_forecasted=_strategy_distributions(
            by_strategy["open_loop_forecasted"],
        ),
        open_loop_forecasted_w_buffer=_strategy_distributions(
            by_strategy["open_loop_forecasted_w_buffer"],
        ),
        mpc_no_buffer=_strategy_distributions(by_strategy["mpc_no_buffer"]),
        mpc_no_buffer_improvement=_improvements(
            by_strategy["open_loop_forecasted"],
            by_strategy["mpc_no_buffer"],
        ),
        curve=_strategy_distributions(by_strategy["curve"]),
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

        _, open_loop_result_df, _ = simulate_control(
            day_optimization_results.base_hour,
            day_optimization_results.previous_control_no_buffer,
            MIN_TEMPERATURE,
            day_optimization_results.control_open_loop_forecasted,
        )
        visualize.plot_simulation_result(
            open_loop_result_df,
            output_directory
            / f"open-loop-forecasted-day{day_optimization_results.day:02d}.png",
            MIN_TEMPERATURE,
        )

        _, open_loop_w_buffer_result_df, _ = simulate_control(
            day_optimization_results.base_hour,
            day_optimization_results.previous_control_w_buffer,
            MIN_TEMPERATURE_W_BUFFER,
            day_optimization_results.control_open_loop_forecasted_w_buffer,
        )
        visualize.plot_simulation_result(
            open_loop_w_buffer_result_df,
            output_directory
            / f"open-loop-forecasted-w-buffer-day{day_optimization_results.day:02d}.png",
            MIN_TEMPERATURE,
        )

        _, mpc_no_buffer_result_df, _ = simulate_control(
            day_optimization_results.base_hour,
            day_optimization_results.previous_control_no_buffer,
            MIN_TEMPERATURE,
            day_optimization_results.control_mpc_no_buffer,
        )
        visualize.plot_simulation_result(
            mpc_no_buffer_result_df,
            output_directory
            / f"mpc-no-buffer-day{day_optimization_results.day:02d}.png",
            MIN_TEMPERATURE,
        )


def compute_mpc_compute_time_statistics(
    data_directory: Path,
    num_days: Optional[int] = None,
) -> None:
    files = sorted(data_directory.glob("day-optimization-results-*.json"))

    if num_days:
        files = files[:num_days]

    times = []
    iterations = []
    successes = []
    stati = []
    messages = []

    for filepath in files:
        with filepath.open() as f:
            day_optimization_results = from_json(DayOptimizationResults, f.read())

        times += day_optimization_results.times_mpc
        iterations += day_optimization_results.iterations_mpc
        successes += day_optimization_results.successes_mpc
        stati += day_optimization_results.stati_mpc
        messages += day_optimization_results.messages_mpc

    logger.info("name\tmin\tmax\tmean\tmeadian\tlen")

    def print_statistics(numbers: Sequence[float], name: str) -> None:
        logger.info(
            f"{name}\t{min(numbers)}\t{max(numbers)}\t{np.mean(numbers)}\t{np.median(numbers)}\t{len(numbers)}"
        )

    print_statistics(times, "times")
    print_statistics(iterations, "iterations")

    logger.info(f"success rate: {sum(successes) / len(successes)}")

    index_max = times.index(max(times))
    logger.info(
        f"the maximum time is {times[index_max]} with {iterations[index_max]} iterations and message '{messages[index_max]}'"
    )

    times.sort()
    iterations.sort()
    logger.debug(f"times sorted: {times}")
    logger.debug(f"iterations sorted: {iterations}")


def compute_overall_energy(
    data_directory: Path,
    num_days: Optional[int] = None,
) -> None:
    files = sorted(data_directory.glob("day-evaluations-*.json"))

    if num_days:
        files = files[:num_days]

    @dataclass
    class ReportedMeasurements:
        heat_fed_in: float = 0
        heat_delivered: float = 0
        heat_loss: float = 0
        pump_energy: float = 0

    @dataclass
    class StrategyMeasurements:
        curve: ReportedMeasurements
        open_loop: ReportedMeasurements
        open_loop_w_buffer: ReportedMeasurements
        mpc: ReportedMeasurements

    measurements = StrategyMeasurements(
        ReportedMeasurements(),
        ReportedMeasurements(),
        ReportedMeasurements(),
        ReportedMeasurements(),
    )

    def add_evaluation(reported: ReportedMeasurements, evaluation: Evaluation) -> None:
        reported.heat_fed_in += evaluation.heat_fed_in
        reported.heat_delivered += evaluation.heat_delivered
        reported.heat_loss += evaluation.heat_loss
        reported.pump_energy += evaluation.pump_energy

    for filepath in files:
        with filepath.open() as f:
            day_evaluations = from_json(DayEvaluations, f.read())

        add_evaluation(measurements.curve, day_evaluations.evaluation_curve)
        add_evaluation(measurements.mpc, day_evaluations.evaluation_mpc_no_buffer)
        add_evaluation(
            measurements.open_loop,
            day_evaluations.evaluation_open_loop_forecasted,
        )
        add_evaluation(
            measurements.open_loop_w_buffer,
            day_evaluations.evaluation_open_loop_forecasted_w_buffer,
        )

    logger.info(
        f"energy in sum:\ncurve: {measurements.curve}\nopen-loop: {measurements.open_loop}\nopen-loop w/ buffer: {measurements.open_loop_w_buffer}\nmpc: {measurements.mpc}"
    )
