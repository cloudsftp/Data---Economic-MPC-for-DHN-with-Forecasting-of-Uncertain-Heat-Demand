from loguru import logger
import numpy as np

from mpc.optimizer.objective import J2
from mpc.simulator import Simulator


def absolute_performance_index(
    simulator: Simulator,
    mpc_control: list[float],
    equilibrium_control: list[float],
) -> float:
    objective = J2().build_objective(simulator)

    mpc_objective = objective(np.array(mpc_control[1:]))
    equilibrium_objective = objective(np.array(equilibrium_control[1:]))
    logger.debug(
        f"objectives: mpc={mpc_objective}, equ={equilibrium_objective} -> diff={mpc_objective - equilibrium_objective}"
    )

    return mpc_objective - equilibrium_objective
