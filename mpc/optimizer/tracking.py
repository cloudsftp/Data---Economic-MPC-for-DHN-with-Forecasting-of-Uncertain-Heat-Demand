from loguru import logger
import numpy as np
from mpc.optimizer.constraints import Constraint, satisfies_all_constraints


class OptimumTracker:
    best_control: np.ndarray
    best_objective_value: float

    constraints: list[Constraint]

    def __init__(
        self,
        initial_control: np.ndarray,
        constraints: list[Constraint],
    ):
        self.best_control = initial_control
        self.best_objective_value = np.inf

        self.constraints = constraints

    def track(
        self,
        control: np.ndarray,
        objective_value: float,
    ):
        logger.debug(f"tracking {control} with value {objective_value}")

        if not np.isfinite(objective_value):
            return
        if not satisfies_all_constraints(control, self.constraints):
            logger.warning("did not satisfy all constraints")
            return
        if objective_value >= self.best_objective_value:
            logger.warning(
                f"no improvement, best objective value: {self.best_objective_value}"
            )
            return

        self.best_control = control.copy()
        self.best_objective_value = objective_value
