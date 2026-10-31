from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Optional
import math
import json

from loguru import logger

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import pandas as pd
import numpy as np

from mpc.time import HOURS
from mpc.water import MIN_TEMPERATURE, temperature

if TYPE_CHECKING:
    from mpc.multi_day import MultiDayDistributions

STEELBLUE = "#005B7F"
FRAUNHOFERGREEN = "#179C7D"
SILVERGREY = "#A6BBC8"
ORANGE = "#F58220"
GRAPHIT = "#1C3F52"
SAND = "#D3C7AE"
PETROL = "#008598"
AQUA = "#39C1CD"
LIME = "#B2D235"
YELLOW = "#FDB913"
RED = "#BB0056"
WINERED = "#7C154D"


def plot_control(
    control: list[float],
    demands: dict[str, list[float]],
    output_file: Path = Path("results/control.png"),
):
    fig, ax1 = plt.subplots(figsize=(12, 5))

    color = "tab:red"
    ax1.set_xlabel("time [h]")
    ax1.set_ylabel("source temperature [°C]", color=color)
    ax1.plot(control, color=color)

    ax2 = ax1.twinx()

    color = "tab:blue"
    ax2.set_ylabel("demand [1]", color=color)
    for name in demands:
        ax2.plot(demands[name], color=color)

    os.makedirs("results", exist_ok=True)
    fig.savefig(output_file)
    plt.close(fig)


def plot_controls(
    controls: list[list[float]],
    demands: dict[str, list[float]],
    output_file: Path = Path("results/control.png"),
):
    fig, ax1 = plt.subplots(figsize=(12, 5))

    color = "tab:red"
    ax1.set_xlabel("time [h]")
    ax1.set_ylabel("source temperature [°C]", color=color)
    for control in controls:
        ax1.plot(control, color=color)

    ax2 = ax1.twinx()

    color = "tab:blue"
    ax2.set_ylabel("demand [1]", color=color)
    for name in demands:
        ax2.plot(demands[name], color=color)

    os.makedirs("results", exist_ok=True)
    fig.savefig(output_file)
    plt.close(fig)


def plot_objective_values(objective_values: list[float]):
    fig, ax = plt.subplots(figsize=(12, 5))

    color = "tab:blue"
    ax.plot(objective_values, color=color)
    ax.set_xlabel("iterations")
    ax.set_ylabel("source energy [Wh]", color=color)

    os.makedirs("results", exist_ok=True)
    fig.savefig("results/objective.png")
    plt.close(fig)


def plot_specific_inner_energy(data: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(12, 5))

    color = "tab:red"
    data["eps"].plot(ax=ax, color=color, label="eps")
    ax.set_xlabel("e [J]")
    ax.set_ylabel("eps [J/kg]")

    color = "tab:blue"
    data["e/rho"].plot(ax=ax, color=color, label="e/rho")

    color = "grey"
    linestyle = "--"
    left = 0.2e9
    right = 0.5e9
    alpha = 0.3
    ax.axvline(left, color=color, linestyle=linestyle)
    ax.axvline(right, color=color, linestyle=linestyle)
    ax.axvspan(left, right, color=color, alpha=alpha)

    fig.legend()

    os.makedirs("results", exist_ok=True)
    fig.savefig("results/water_specific_inner_energy.png")
    plt.close(fig)


FEED_TEMPERATURE_COLORS = [RED, AQUA, FRAUNHOFERGREEN, GRAPHIT]
CONSUMER_DEMAND_COLORS = [AQUA, FRAUNHOFERGREEN, GRAPHIT]


def plot_simulation_result(
    results: pd.DataFrame,
    output_file: Path = Path("results/simulation_temperatures.png"),
    min_temperature: float = MIN_TEMPERATURE,
    visualization_duration: Optional[float] = None,
):
    fig, ax1 = plt.subplots(figsize=(12, 5))

    if visualization_duration is not None:
        end_time = results.index.min() + visualization_duration * HOURS
        results = results.loc[results.index <= end_time]

    time_hours = results.index.to_numpy(dtype=float) / HOURS

    start_hour = int(math.floor(time_hours[0]))
    end_hour = int(math.ceil(time_hours[-1]))
    ticks = range(start_hour, end_hour + 1)
    tick_labels = [str(x) for x in ticks]
    ax1.set_xticks(ticks, tick_labels)
    ax1.set_xlim((start_hour - 0.5, end_hour + 0.5))

    ax1.set_xlabel("time [h]")
    ax1.xaxis.set_major_locator(MultipleLocator(1))

    ax1.set_ylabel("feed temperature [°C]")
    ax1.set_ylim(69, 79)
    energy_density_columns = [
        "S0.1.e",
        "C3.0.e",
        "C4.0.e",
        "C5.0.e",
    ]
    feed_temperatures = temperature(results[energy_density_columns])
    for column, color in zip(energy_density_columns, FEED_TEMPERATURE_COLORS):
        label = f"feed temperature {column.split(".")[0]}"
        ax1.plot(time_hours, feed_temperatures[column], color=color, label=label)

    ax1.axhline(y=min_temperature, color=ORANGE)

    temperature_C5 = feed_temperatures["C5.0.e"]
    ax1.fill_between(
        time_hours,
        temperature_C5,
        min_temperature,
        where=temperature_C5 < min_temperature,  # type: ignore
        color=ORANGE,
        alpha=0.5,
        interpolate=True,
    )
    # ax1.fill_between(
    #     time_hours,
    #     temperature_C5,
    #     min_temperature,
    #     where=temperature_C5 > min_temperature,  # type: ignore
    #     color=FRAUNHOFERGREEN,
    #     alpha=0.5,
    #     interpolate=True,
    # )

    ax2 = ax1.twinx()

    ax2.set_ylabel("consumer demand [kW]")
    consumer_names = ["C3", "C4", "C5"]
    for name, color in zip(consumer_names, CONSUMER_DEMAND_COLORS):
        power = (results[f"{name}.0.Q"] + results[f"{name}.1.Q"]) / 1000
        ax2.plot(
            time_hours,
            power,
            color=color,
            alpha=0.5,
            label=f"heat demand {name}",
            linestyle="--",
        )

    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file)
    plt.close(fig)


def plot_horizon_performance_indices(
    horizons: list[int],
    distances: list[float],
    times: list[float],
    output_figure_file: Path,
):
    fig, ax1 = plt.subplots(figsize=(12, 5))

    ax1.set_xticks([2 * (i + 1) for i in range(12)])
    ax1.set_xlabel("prediction horizon [h]")

    ax1.plot(
        horizons,
        distances,
        color=AQUA,
        marker="s",
        label="absolute performance index",
    )
    ax1.set_ylabel("performance index")
    ax1.set_ylim(-5.0e1, -1.2e3)
    ax1.yaxis.set_inverted(True)
    ax1.set_yscale("symlog")

    ax2 = ax1.twinx()

    ax2.plot(
        horizons,
        times,
        color=FRAUNHOFERGREEN,
        marker="s",
        label="median optimization time",
    )
    ax2.set_ylabel("optimization time [s]")

    fig.legend()

    output_figure_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_figure_file)
    plt.close(fig)


def plot_evaluation_distributions(
    distributions: MultiDayDistributions,
    output_figure_file: Path,
) -> None:
    strategies = [
        ("temperature\ncurve", distributions.curve),
        ("open-\nloop", distributions.open_loop_forecasted),
        # ("open-loop\nw/ buffer", distributions.open_loop_forecasted_w_buffer),
        ("NMPC", distributions.mpc_no_buffer),
    ]
    metrics = [
        (
            "Heat loss",
            "heat_loss",
            "energy [Wh]",
            6,
        ),
        (
            "Pump energy",
            "pump_energy",
            "energy [Wh]",
            3,
        ),
        (
            "Maximum violation",
            "maximum",
            "temperature [K]",
            0,
        ),
        # (
        #    "Violation area",
        #    "area",
        #    "temperature with time [Kh]",
        #    0,
        # ),
    ]

    fig, axes = plt.subplots(1, len(metrics), figsize=(12, 5), constrained_layout=True)

    for ax, (title, attribute, ylabel, texp) in zip(axes, metrics):
        stats = []
        for label, strategy_distributions in strategies:
            distribution = getattr(strategy_distributions, attribute)
            stats.append(
                {
                    "label": label,
                    "whislo": distribution.minimum,
                    "q1": distribution.q1,
                    "med": distribution.median,
                    "q3": distribution.q3,
                    "whishi": distribution.maximum,
                    "mean": distribution.mean,
                    "fliers": [],
                }
            )

        logger.info(f"{title}:\n{json.dumps(stats, indent=2)}")

        boxplot = ax.bxp(stats, showmeans=True, showfliers=False, patch_artist=True)
        for box in boxplot["boxes"]:
            box.set(facecolor=AQUA, alpha=0.6, edgecolor=GRAPHIT)
        for median in boxplot["medians"]:
            median.set(color=GRAPHIT, linewidth=1.5)
        for mean in boxplot["means"]:
            mean.set(
                marker="o",
                markerfacecolor=ORANGE,
                markeredgecolor=GRAPHIT,
                markersize=5,
            )

        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x")
        ax.grid(axis="y", color=SILVERGREY, alpha=0.35)

        ax.ticklabel_format(axis="y", style="sci", scilimits=(texp, texp))

    output_figure_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_figure_file)
    plt.close(fig)


def plot_demands(
    demands: dict[str, list[float]],
    predicted_demands: dict[str, list[float]],
    filename: str,
):
    fig, ax1 = plt.subplots(figsize=(12, 5))

    ax1.set_xlabel("time [h]")
    ax1.set_ylabel("demand [1]")

    logger.debug(f"demands: {demands}")

    for con in demands:
        ax1.plot(demands[con])
        ax1.plot(predicted_demands[con])

    fig.savefig(filename)
    plt.close(fig)


def plot_simulation_result_not(
    parameters: np.ndarray,
    previous_demands: dict[str, list[float]],
    filename: str = "results/simulation_result.png",
) -> None:
    """
    Plot the control parameters together with the previous real demands
    for each consumer (C1..C5).

    Left y-axis: source/control temperature (parameters)
    Right y-axis: previous demands per consumer
    """
    fig, ax1 = plt.subplots()

    # --- Control / parameters (left axis) ---
    color = "tab:red"
    ax1.set_xlabel("time step")
    ax1.set_ylabel("source temperature [°C]", color=color)
    ax1.plot(parameters, color=color, label="control (parameters)")
    ax1.tick_params(axis="y", labelcolor=color)

    # --- Previous demands (right axis) ---
    ax2 = ax1.twinx()
    ax2.set_ylabel("demand [kWh]", color="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:blue")

    for name in sorted(previous_demands.keys()):
        demand = previous_demands[name]
        ax2.plot(demand, label=f"{name} previous demand", linewidth=1.0)

    fig.legend(loc="upper right")

    os.makedirs("results", exist_ok=True)
    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)
