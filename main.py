from pathlib import Path
import json
from typing import Optional

import typer
from loguru import logger

from mpc import multi_day, visualize
from mpc.config import (
    DEFAULT_NETWORK_DIRECTORY,
    CONSUMER_DATA_DIRECOTRY,
    CLUSTER_CONFIG,
    FutureDataMode,
    get_demand_provider,
)
from mpc.curve import Curve
from mpc.demand import DemandProvider, DemandProviderI, HistoricalDemandProvider
from mpc.mpc import MPC
from mpc.optimizer.objective import J1
from mpc.optimizer import Optimizer, OptimizerUpdate
from mpc.simulator import Simulator
from mpc.simulator.parsing import _read_output_csv
from mpc.time import HOURS
from mpc.water import MIN_TEMPERATURE, MIN_TEMPERATURE_W_BUFFER
from mpc.results import load_mpc_results

app = typer.Typer()


PREVIOUS_CONTROL = [77.0] * 13 + [
    75.59329574862358,
    75.84187185772657,
    74.12224012882426,
    75.20254432571609,
    75.06736570878685,
    74.27539467307442,
    73.95366143653631,
    74.15502131486286,
    74.3408332309498,
    73.94896429432866,
    73.96577962623587,
    74.32453886521752,
]

PREVIOUS_CONTROL_W_BUFFER = [77.0] * 13 + [
    76.20420763830052,
    76.04943203751387,
    74.58571867848505,
    75.65025538033258,
    75.45782688941122,
    74.65111623991564,
    74.28307747292547,
    74.62719035972779,
    74.64364498912842,
    74.31609197671906,
    74.33966846785417,
    74.7672363791095,
]

PREVIOUS_CONTROL_CURVE = [88.875] * 3 + [92.0] * 21


@app.command()
def run_open_loop(
    horizon_hours: int = 24,
    mode: FutureDataMode = FutureDataMode.Forecasted,
):
    num_controls = horizon_hours

    # previous_control = [77.0] * (horizon_hours + 1)
    previous_control = PREVIOUS_CONTROL
    # previous_control = PREVIOUS_CONTROL_W_BUFFER
    base_hour = (
        15740  # 14273 # starting index in test data # 14762, 15251, 15740, 16229, 16718
        # - 12
    )
    demand_provider = get_demand_provider(mode, base_hour, horizon_hours)

    demand_provider.assert_enough_data_available(horizon_hours)
    previous_demands = demand_provider.get_previous_demands()
    demands = demand_provider.get_predicted_future_demands()

    initial_control = [77.0] * num_controls

    objective_values = []

    def visualize_progress(update: OptimizerUpdate):
        visualize.plot_control(update.current_control, demands)

        objective_values.append(update.current_objective_value)
        visualize.plot_objective_values(objective_values)

        # logger.debug(f"initial energy density: {update.simulation_result["S0.1.e"][0]}")
        visualize.plot_simulation_result(
            update.simulation_result,
        )

    opt = Optimizer(
        DEFAULT_NETWORK_DIRECTORY,
        J1(),
        previous_control,
        previous_demands,
        initial_control,
        demands,
        horizon_hours,
        MIN_TEMPERATURE,
        report_intermediate_result=visualize_progress,
        hours_ensure_constraints=4,
    )

    optimization_result = opt.optimize()

    logger.info(
        f"success: {optimization_result.optimized}, optimal control: {optimization_result.control}"
    )


@app.command()
def run_mpc(
    horizon_hours: int = 8,
    num_steps: int = 24,
    mode: FutureDataMode = FutureDataMode.Forecasted,
    state_update_hours: int = 1,
    prediction_update_hours: int = 1,
):
    if state_update_hours > prediction_update_hours:
        raise RuntimeError(
            "time to update the state must be smaller or equal to the time to update the predictions"
        )

    num_controls = horizon_hours

    base_hour = (
        15740  # 14273 # starting index in test data # 14762, 15251, 15740, 16229, 16718
    )

    demand_provider = get_demand_provider(
        mode,
        base_hour,
        horizon_hours,
        prediction_update_hours,
    )

    mpc = MPC(
        DEFAULT_NETWORK_DIRECTORY,
        J1(),
        demand_provider,
        PREVIOUS_CONTROL,
        initial_control_temperature=77.0,
        num_controls=num_controls,
        horizon_hours=horizon_hours,
        num_steps=num_steps,
        min_temperature=MIN_TEMPERATURE,
        state_update_hours=state_update_hours,
    )
    mpc_results = mpc.run()

    logger.info(f"MPC finished:\n\n{mpc_results}")
    with open("results/optimal_control.json", "w") as f:
        json.dump(mpc_results.__dict__, f, indent=4)


@app.command()
def run_curve():
    base_hour = 15740  # - 24

    curve = Curve(
        base_hour,
    )

    logger.info(f"control for hour {base_hour}: {curve.control()}")


multi_day_data_directory = Path("data/results/multi-day")


@app.command()
def run_multiple_day_experiment(
    num_days: int,
    start_day_offset: int = 0,
    skip_opt: bool = False,
):
    if skip_opt:
        logger.warning("Skipping optimizations")

    multi_day.run_experiments(
        15740,
        start_day_offset,
        num_days,
        multi_day_data_directory,
        skip_opt,
    )


@app.command()
def evaluate_multiple_day_experiment(num_days: Optional[int] = None):
    multi_day.evaluate_optimization_results(
        multi_day_data_directory,
        num_days,
    )


@app.command()
def plot_multiple_day_experiment_evaluation_distributions(
    num_days: Optional[int] = None,
):
    multi_day.plot_evaluation_distributions(
        multi_day_data_directory,
        Path(
            "Paper/02_Sections/05_Results/img/Aroma/Results/evaluation-statistics.png"
        ),
        num_days,
    )


@app.command()
def compute_multiple_day_experiment_overall_energy(
    num_days: Optional[int] = None,
):
    multi_day.compute_overall_energy(multi_day_data_directory, num_days)


@app.command()
def plot_multiple_day_experiment_simulations(num_days: Optional[int] = None):
    multi_day.simulate_and_plot_results(
        multi_day_data_directory,
        Path("results/multi-day"),
        num_days,
    )


@app.command()
def compute_mpc_compute_time_statistics(num_days: Optional[int] = None) -> None:
    multi_day.compute_mpc_compute_time_statistics(
        multi_day_data_directory,
        num_days,
    )


@app.command()
def generate_horizon_comparison_images():
    horizon_hours = 8

    base_hour = (
        15740  # 14273 # starting index in test data # 14762, 15251, 15740, 16229, 16718
    )

    demand_provider = get_demand_provider(
        FutureDataMode.Actual, base_hour, horizon_hours
    )

    simulator = Simulator(
        DEFAULT_NETWORK_DIRECTORY,
        PREVIOUS_CONTROL,
        demand_provider.get_previous_demands(),
        horizon_hours * HOURS,
        0,
        0,
        MIN_TEMPERATURE,
    )

    mpc_results = load_mpc_results(
        Path("data/results/forecast-no-buffer-12-steps-slsqp-ftol+10")
    )
    horizons = []
    distances = []
    times = []

    equilibrium_control = [
        74.32453886521752,
        74.1411830739982,
        74.05004615327769,
        74.35170965175763,
        73.93648163112938,
        73.79370095657143,
        73.63244937735959,
        73.21975216346324,
        73.26678125428737,
        73.13954341839907,
        73.297332860769,
        74.25461516805711,
        74.90267837876421,
        74.78266337717221,
        75.34356185446124,
        75.19452144396747,
        74.55437261769904,
        74.25903475813686,
        74.5272671925361,
        74.59759136350335,
        74.06544936815854,
        74.59812161992501,
        74.42664414635955,
        74.58836584037499,
        74.48949917897608,
        74.23905152345765,
    ]
    equilibrium_control = equilibrium_control[:13]

    simulator.fixed_control[-1] = equilibrium_control[1]
    simulator.set_demands(demand_provider.get_predicted_future_demands())

    for result in mpc_results:
        logger.debug(f"result: {result.result.optimal_control}")
        performance_index = -result.absolute_performance_index(
            simulator,
            equilibrium_control,
        )

        horizons.append(result.horizon)
        distances.append(performance_index)
        times.append(result.median_time())

        simulator.set_controls(result.result.optimal_control)
        simulator._simulate()
        results_file = Path("/tmp/simulation-results.csv")
        simulator._export_outputs(results_file)
        results_df = _read_output_csv(results_file)
        visualize.plot_simulation_result(
            results_df, Path(f"/tmp/horizon-{result.horizon}.png")
        )

        logger.info(
            f"horizon {result.horizon} compared to equilibrium: {performance_index}"
        )

    visualize.plot_horizon_performance_indices(
        horizons,
        distances,
        times,
        Path(
            "Paper/02_Sections/05_Results/img/Aroma/CompareHorizons/horizon-performances.png"
        ),
    )


@app.command()
def debug_mpc_step(
    step: int,
    optimization_output_file: Path = Path("results/optimal_control.json"),
    horizon_hours: int = 8,
    prediction_update_hours: int = 1,
    mode: FutureDataMode = FutureDataMode.Forecasted,
):
    num_controls = horizon_hours

    with open(optimization_output_file) as f:
        mpc_results = json.load(f)
        fixed_control = [mpc_results["optimal_control"][step]]

    previous_control = [74.0] * (num_controls - len(fixed_control)) + fixed_control

    base_hour = (
        15740  # 14273 # starting index in test data # 14762, 15251, 15740, 16229, 16718
    ) + step

    match mode:
        case FutureDataMode.Forecasted:
            demand_provider: DemandProviderI = DemandProvider(
                CONSUMER_DATA_DIRECOTRY,
                CLUSTER_CONFIG,
                base_hour,
                horizon_hours,
                prediction_update_hours,
            )
        case FutureDataMode.Actual:
            demand_provider = HistoricalDemandProvider(
                CONSUMER_DATA_DIRECOTRY,
                CLUSTER_CONFIG,
                base_hour,
                horizon_hours,
            )

    initial_control = [75.0] * num_controls

    previous_demands = demand_provider.get_previous_demands()
    demands = demand_provider.get_predicted_future_demands()

    objective_values = []

    def visualize_progress(update: OptimizerUpdate):
        visualize.plot_control(update.current_control, demands)

        objective_values.append(update.current_objective_value)
        visualize.plot_objective_values(objective_values)

        # logger.debug(f"initial energy density: {update.simulation_result["S0.1.e"][0]}")
        visualize.plot_simulation_result(update.simulation_result)

    opt = Optimizer(
        DEFAULT_NETWORK_DIRECTORY,
        J1(),
        previous_control,
        previous_demands,
        initial_control,
        demands,
        horizon_hours,
        MIN_TEMPERATURE,
        report_intermediate_result=visualize_progress,
        hours_ignore_temp_constraints=4,
    )

    opt._simulator._iteration = step
    opt._simulator.set_simulation_time_for_simulation()
    opt._simulator.set_demands(demands)

    optimization_result = opt.optimize()

    logger.info(
        f"success: {optimization_result.optimized}, optimal control: {optimization_result.control}"
    )


if __name__ == "__main__":
    logger.add("optimization.log", level="TRACE", mode="w")

    app()
