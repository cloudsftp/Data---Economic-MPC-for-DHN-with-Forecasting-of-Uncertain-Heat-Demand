from dataclasses import dataclass
from pathlib import Path

from loguru import logger
import numpy as np
from scipy import integrate

from main import PREVIOUS_CONTROL_CURVE, PREVIOUS_CONTROL_W_BUFFER, PREVIOUS_CONTROL
from mpc import visualize
from mpc.demand import DemandProvider
from mpc.profiles.multi_consumer_forecast import (
    ClusterConfig,
)
from mpc.simulator import HOURS, Simulator
from mpc.simulator.parsing import evaluate_result_file, _read_output_csv
from mpc.water import MIN_TEMPERATURE, temperature

C5_FEED_ENERGY_DENSITY_COLUMN = "C5.0.e"


@dataclass
class ControlConfig:
    control: list[float]
    previous_control: list[float]


optimal_controls = {
    "mpc": ControlConfig(
        [
            74.83161143385757,
            74.22214432445554,
            73.96465976114091,
            73.59036452819502,
            73.23874340612846,
            73.34968341638807,
            73.36160566864038,
            73.46450751667655,
            73.74858900525061,
            73.92163750226926,
            74.18287142384747,
            74.25241999281826,
            74.37296395468523,
            75.26907351133819,
            74.51498285118026,
            74.82610645517128,
            74.670313970024,
            74.59507150102218,
            74.59117722256615,
            74.35390914834454,
            74.42642503132427,
            74.66378231539,
            74.41506609283076,
            74.60371824672197,
        ],
        PREVIOUS_CONTROL,
    ),
    "mpc-01-01-24": ControlConfig(
        [
            74.82198918554315,
            74.22370853390285,
            73.96368564400682,
            73.59061727434445,
            73.23851280015874,
            73.34852412860761,
            73.36154784584058,
            73.46459565392239,
            73.74857800254995,
            73.92158272240705,
            74.1828721315629,
            74.2524200384461,
            74.37297574444595,
            75.26995838082027,
            74.51467484983212,
            74.82765339174485,
            74.66771560638192,
            74.59888954439026,
            74.59115433758784,
            74.3548228206789,
            74.42691161385648,
            74.6630944010648,
            74.41579608942591,
            74.60274627067875,
        ],
        PREVIOUS_CONTROL,
    ),
    "mpc-01-24-08": ControlConfig(
        [
            74.83161143385757,
            74.21256192402683,
            74.11310092079793,
            73.59991365657233,
            73.28537126649215,
            73.40527212386736,
            73.45404151568297,
            73.61785648838183,
            73.75988677218534,
            73.89917369641013,
            74.18495719668084,
            74.5001167128742,
            74.38140806888174,
            74.15585381438146,
            74.01775965938857,
            73.93830632188548,
            73.98142670349186,
            73.97587287598088,
            73.943050905865,
            73.9836418855584,
            74.01203289784176,
            74.08054279205194,
            74.06331416096333,
            74.05108489812547,
        ],
        PREVIOUS_CONTROL,
    ),
    "open-forecast": ControlConfig(
        [
            74.82199606926751,
            74.21320131442386,
            74.110465211942,
            73.59783813842229,
            73.29864173537825,
            73.40982132529923,
            73.45966768601723,
            73.62336233191007,
            73.76520690800365,
            73.89268027479802,
            74.17970453729109,
            74.49388411591289,
            74.37222956501387,
            74.15805935664874,
            74.02179733167904,
            73.9495100458418,
            73.99193291221329,
            73.99679459743074,
            73.95552125141688,
            73.99163126324882,
            74.021171579682,
            74.08853052386506,
            74.0642326805851,
            74.05568433695021,
        ],
        PREVIOUS_CONTROL,
    ),
    "open-forecast-w-buffer": ControlConfig(
        [
            75.2168461686519,
            74.62125534497372,
            74.42802828848832,
            73.93750043772351,
            73.6583167758297,
            73.77333290670636,
            73.81990779233712,
            73.98976527076147,
            74.13606268052946,
            74.26696394919425,
            74.57592238525784,
            74.88212024795374,
            74.74343661029484,
            74.52490016509013,
            74.39400876384211,
            74.31961539081561,
            74.3642522921074,
            74.36815646373047,
            74.3272250096568,
            74.36601361235297,
            74.39113280767988,
            74.47036806797036,
            74.43235560175734,
            74.49326722725588,
        ],
        PREVIOUS_CONTROL_W_BUFFER,
    ),
    "open-actual": ControlConfig(
        [
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
        ],
        PREVIOUS_CONTROL,
    ),
    "curve": ControlConfig(
        [92.0] * 3 + [90.25] * 21,
        PREVIOUS_CONTROL_CURVE,
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
    prediction_update_hours=1,
)

actual_demands = demand_provider.get_previous_demands()

data_directory = Path("data")
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
        0,
        MIN_TEMPERATURE,
    )
    simulator.set_demands(future_demands)

    simulator.set_controls(control_config.control)
    simulator._simulate()
    result_file = data_directory / "compare" / f"{name}.csv"
    result_file.parent.mkdir(parents=True, exist_ok=True)
    simulator._export_outputs(Path(f"../compare/{name}.csv"))

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
simulate_and_visualize("open-forecast-w-buffer")

simulate_and_visualize("mpc")
simulate_and_visualize("mpc-01-01-24")
simulate_and_visualize("mpc-01-24-08")
simulate_and_visualize("curve")
simulate_and_visualize("full-send")
