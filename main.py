from pathlib import Path
import json
from typing import Optional

import typer
from loguru import logger
import pandas as pd
import numpy as np

from mpc import multi_day, performance, visualize
from mpc.config import (
    DEFAULT_NETWORK_DIRECTORY,
    CONSUMER_DATA_DIRECOTRY,
    CLUSTER_CONFIG,
    FutureDataMode,
    get_demand_provider,
)
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


PREVIOUS_CONTROL = [74.0] * 13 + [
    73.8538439501464,
    73.9667852916009,
    72.7669168193153,
    73.53854978473106,
    73.43470259601197,
    72.92317066649865,
    72.6874470190878,
    72.83074695039944,
    72.95918761045886,
    72.70406764080107,
    72.70310966510026,
    72.97729540293285,
]

PREVIOUS_CONTROL_W_BUFFER = [74.0] * 13 + [
    74.3050643820824,
    74.2674916444755,
    73.06291091048975,
    73.95145315654912,
    73.76837730663593,
    73.23494112913355,
    73.01578886875534,
    73.1709356925878,
    73.30367704429763,
    73.02625597756116,
    73.04416585057722,
    73.34222503068223,
]


@app.command()
def run_open_loop(
    horizon_hours: int = 24,
    mode: FutureDataMode = FutureDataMode.Forecasted,
):
    num_controls = horizon_hours

    # previous_control = [74.0] * (horizon_hours + 1)
    previous_control = PREVIOUS_CONTROL
    base_hour = (
        15740  # 14273 # starting index in test data # 14762, 15251, 15740, 16229, 16718
        # - 12
    )
    demand_provider = get_demand_provider(mode, base_hour, horizon_hours)

    demand_provider.assert_enough_data_available(horizon_hours)
    previous_demands = demand_provider.get_previous_demands()
    demands = demand_provider.get_predicted_future_demands()

    initial_control = [75.0] * num_controls

    objective_values = []

    def visualize_progress(update: OptimizerUpdate):
        visualize.plot_control(update.current_control, demands)

        objective_values.append(update.current_objective_value)
        visualize.plot_objective_values(objective_values)

        # logger.debug(f"initial energy density: {update.simulation_result["S0.1.e"][0]}")
        visualize.plot_simulation_result(
            update.simulation_result,
            visualization_duration=24,
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
):
    num_controls = horizon_hours

    base_hour = (
        15740  # 14273 # starting index in test data # 14762, 15251, 15740, 16229, 16718
    )

    demand_provider = get_demand_provider(mode, base_hour, horizon_hours)

    mpc = MPC(
        DEFAULT_NETWORK_DIRECTORY,
        J1(),
        demand_provider,
        PREVIOUS_CONTROL,
        initial_control_temperature=75.0,
        num_controls=num_controls,
        horizon_hours=horizon_hours,
        num_steps=num_steps,
        min_temperature=MIN_TEMPERATURE,
    )
    mpc_results = mpc.run()

    logger.info(f"MPC finished:\n\n{mpc_results}")
    with open("results/optimal_control.json", "w") as f:
        json.dump(mpc_results.__dict__, f, indent=4)


multi_day_data_directory = Path("data/results/multi-day")


@app.command()
def run_multiple_day_experiment(num_days: int, start_day_offset: int = 0):
    multi_day.run_experiments(
        15740,
        start_day_offset,
        num_days,
        multi_day_data_directory,
    )


@app.command()
def evaluate_multiple_day_experiment(num_days: Optional[int] = None):
    multi_day.evaluate_optimization_results(
        Path("data/results/multi-day"),
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
def plot_multiple_day_experiment_simulations(num_days: Optional[int] = None):
    multi_day.simulate_and_plot_results(
        multi_day_data_directory,
        Path("results/multi-day"),
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
        MIN_TEMPERATURE,
    )

    mpc_results = load_mpc_results(
        Path("data/results/actual-no-buffer-12-steps-slsqp-ftol+10")
    )
    horizons = []
    distances = []
    times = []

    equilibrium_control = [
        72.97729540293285,
        72.81990957158403,
        72.77459412126079,
        72.97449361077638,
        72.69471625123083,
        72.6046029234796,
        72.4923600047734,
        72.20472027549093,
        72.24457005616813,
        72.15330089665109,
        72.25840896648536,
        72.91136032277879,
        73.28436455696001,
        73.29817349524096,
    ]
    equilibrium_control = equilibrium_control[:13]

    simulator.fixed_control[-1] = equilibrium_control[1]
    simulator.set_demands(demand_provider.get_predicted_future_demands())

    for result in mpc_results:
        logger.debug(f"result: {result.result.optimal_control}")
        performance_index = result.absolute_performance_index(
            simulator,
            equilibrium_control,
        )

        horizons.append(result.horizon)
        distances.append(performance_index)
        times.append(result.average_time())

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
