from dataclasses import dataclass
from pathlib import Path

from loguru import logger
import numpy as np
from scipy import integrate

from mpc import visualize
from mpc.demand import DemandProvider
from mpc.profiles.multi_consumer_forecast import (
    ClusterConfig,
)
from mpc.simulator import HOURS, Simulator
from mpc.simulator.parsing import evaluate_result_file, _read_output_csv
from mpc.water import MIN_TEMPERATURE, temperature

C5_FEED_ENERGY_DENSITY_COLUMN = "C5.0.e"

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


@dataclass
class ControlConfig:
    control: list[float]
    previous_control: list[float]


optimal_controls = {
    "mpc": ControlConfig(
        [
            73.24474007248504,
            72.8912825399692,
            72.70756777770356,
            72.46761936436631,
            72.21868715332437,
            72.29521122407314,
            72.30053102129936,
            72.3708481509138,
            72.55575072249626,
            72.67071350942709,
            72.84585416388198,
            72.88988690394483,
            72.95399403796222,
            73.58865321668226,
            73.08317444521393,
            73.26485893131384,
            73.19545769573678,
            73.10460271965654,
            73.12460864086596,
            72.94136642953055,
            73.01695779181523,
            73.15744868874613,
            73.0068820229054,
            73.1277366748635,
        ],
        PREVIOUS_CONTROL,
    ),
    "open-forecast": ControlConfig(
        [
            73.24047905252145,
            72.88811308336611,
            72.79420575268564,
            72.47075287708775,
            72.26103305946496,
            72.33498563973309,
            72.36542019571007,
            72.47311777448662,
            72.56648046751752,
            72.65029442854583,
            72.83547538888214,
            73.05151047033965,
            72.97369273422538,
            72.83427154819199,
            72.74053962184553,
            72.69325980253849,
            72.71925418912444,
            72.72148824986566,
            72.6928735672382,
            72.71784031420943,
            72.73847519062076,
            72.78780086420939,
            72.77113210534209,
            72.76324227393563,
        ],
        PREVIOUS_CONTROL,
    ),
    "open-actual": ControlConfig(
        [
            72.81705823,
            72.77492866,
            72.97424604,
            72.69477968,
            72.60453798,
            72.49230801,
            72.20468819,
            72.24472755,
            72.15306933,
            72.25937898,
            72.88652888,
            73.33920666,
            73.24309864,
            73.62689437,
            73.52829803,
            73.11211968,
            72.90522687,
            73.06381706,
            73.16068283,
            72.76192238,
            73.13717245,
            73.02221262,
            73.0612408,
            73.17882221,
        ],
        PREVIOUS_CONTROL,
    ),
    "full-send": ControlConfig(
        [90.0] * 24,
        [90.0] * 25,
    ),
}

CONSUMER_DATA_DIRECOTRY: Path = Path("data/Profiles")
# Konfiguration: pro Verbraucher Feature- und Consumption-Datei + Modelldatei
CLUSTER_CONFIG: dict[str, ClusterConfig] = {
    "C1": ClusterConfig(
        data_file_name="aggregated_SFH_alt_5_1_features+lags.csv",
        models_file_name="lightgbm_models_aggregated_SFH_alt_5_1_features+lags.pkl",
    ),
    "C2": ClusterConfig(
        data_file_name="aggregated_SFH_alt_5_2_features+lags.csv",
        models_file_name="lightgbm_models_aggregated_SFH_alt_5_2_features+lags.pkl",
    ),
    "C3": ClusterConfig(
        data_file_name="aggregated_SFH_alt_5_3_features+lags.csv",
        models_file_name="lightgbm_models_aggregated_SFH_alt_5_3_features+lags.pkl",
    ),
    "C4": ClusterConfig(
        data_file_name="aggregated_SFH_alt_5_4_features+lags.csv",
        models_file_name="lightgbm_models_aggregated_SFH_alt_5_4_features+lags.pkl",
    ),
    "C5": ClusterConfig(
        data_file_name="aggregated_SFH_alt_5_5_features+lags.csv",
        models_file_name="lightgbm_models_aggregated_SFH_alt_5_5_features+lags.pkl",
    ),
}

base_hour = (
    15740  # 14273 # starting index in test data # 14762, 15251, 15740, 16229, 16718
)
demand_provider = DemandProvider(
    CONSUMER_DATA_DIRECOTRY,
    CLUSTER_CONFIG,
    base_hour,
    window=24,
)

actual_demands = demand_provider.get_previous_demands()

network_directory = Path("data/aroma_mpc")
previous_demands = demand_provider.get_previous_demands()
future_demands = demand_provider.get_actual_future_demands()


def simulate_and_visualize(name: str) -> None:
    control_config = optimal_controls[name]

    simulator = Simulator(
        network_directory,
        control_config.previous_control,
        previous_demands,
        24 * HOURS,
        0,
        MIN_TEMPERATURE,
    )
    simulator.set_demands(future_demands)

    simulator.set_controls(control_config.control)
    simulator._simulate()
    result_file = network_directory / "compare" / f"{name}.csv"
    result_file.parent.mkdir(parents=True, exist_ok=True)
    simulator._export_outputs(Path(f"compare/{name}.csv"))

    results = evaluate_result_file(
        result_file,
        simulator.num_time_steps,
        1,
        24,
        0,
        0,
        MIN_TEMPERATURE,
        "C5",
    )
    logger.info(f"{name}: {results.objective}")

    results_df = _read_output_csv(result_file)

    # compute violation metrics

    time_hours = results_df.index.to_numpy(dtype=float) / HOURS
    feed_temperature_C5 = temperature(results_df[C5_FEED_ENERGY_DENSITY_COLUMN])
    deficit = np.maximum(MIN_TEMPERATURE - feed_temperature_C5, 0)

    area = np.trapezoid(deficit, x=time_hours)
    max_depth = deficit.max()
    duration = np.trapezoid(deficit > 0, x=time_hours)

    logger.info(f"violation metrics: area={area}, max={max_depth}, duration={duration}")

    # compute energy loss

    columns = list(map(str, results_df.columns))

    def extract_name(prefix: str) -> list[str]:
        return list(
            set(
                map(
                    lambda v: v.split(".")[0],
                    filter(
                        lambda v: v.startswith(prefix),
                        columns,
                    ),
                )
            )
        )

    consumers = extract_name("C")
    sources = extract_name("S")

    logger.debug(f"consumers: {consumers}")
    logger.debug(f"sources: {sources}")

    loss = 0.0
    total = 0.0

    def add_energy(name: str):
        nonlocal loss, total

        Qin = name + ".0.Q"
        Qout = name + ".1.Q"

        energy = integrate.trapezoid(
            results_df[Qin] + results_df[Qout],
            dx=0.25,
        )

        loss += energy
        if name.startswith("S"):
            total += energy

    for consumer in consumers:
        add_energy(consumer)

    for source in sources:
        add_energy(source)

    logger.info(f"energy lost: {loss} (total: {total} -> {loss/total*100} %)")

    # plot simulation

    result_file = (
        Path("Paper/02_Sections/05_Results/img/Aroma/Results")
        / f"{name}-evaluation.png"
    )
    result_file.parent.mkdir(parents=True, exist_ok=True)
    visualize.plot_simulation_result(
        results_df,
        result_file,
    )


simulate_and_visualize("open-actual")
simulate_and_visualize("open-forecast")

simulate_and_visualize("mpc")
simulate_and_visualize("full-send")
