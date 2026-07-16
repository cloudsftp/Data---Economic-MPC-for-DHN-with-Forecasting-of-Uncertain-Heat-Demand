import numpy as np
from scipy.optimize import LinearConstraint, Bounds

MAX_CHANGE_IN_TEMPERATURE = 4.0

MIN_TEMPERATURE = 70.0
MAX_TEMPERATURE = 95.0


def _changes_in_temperature_matrix(n: int) -> np.ndarray:
    rows = []

    for i in range(n - 1):
        row = [0.0] * n

        row[i] = -1.0
        row[i + 1] = 1.0

        rows.append(row)

    matrix = np.array(rows)
    return matrix


def changes_in_temperature_constraint(
    n: int,
) -> LinearConstraint:
    matrix = _changes_in_temperature_matrix(n)

    lb = -MAX_CHANGE_IN_TEMPERATURE * np.ones(n - 1)
    ub = MAX_CHANGE_IN_TEMPERATURE * np.ones(n - 1)

    return LinearConstraint(matrix, lb, ub)


def boundaries(
    n: int,
    last_fixed_control_value: float,
) -> Bounds:

    lower_first_bound = max(
        last_fixed_control_value - MAX_CHANGE_IN_TEMPERATURE,
        MIN_TEMPERATURE,
    )
    lb = np.concatenate(
        (
            np.array([lower_first_bound]),
            MIN_TEMPERATURE * np.ones(n - 2),
            np.array([lower_first_bound]),
        )
    )

    upper_first_bound = min(
        last_fixed_control_value + MAX_CHANGE_IN_TEMPERATURE,
        MAX_TEMPERATURE,
    )

    ub = np.concatenate(
        (
            np.array([upper_first_bound]),
            MAX_TEMPERATURE * np.ones(n - 2),
            np.array([upper_first_bound]),
        )
    )

    return Bounds(lb, ub, keep_feasible=False)
