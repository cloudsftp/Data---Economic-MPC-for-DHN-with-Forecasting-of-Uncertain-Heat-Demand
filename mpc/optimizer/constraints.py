import numpy as np
from scipy.optimize import (
    Bounds,
    LinearConstraint,
    NonlinearConstraint,
)

CONSTRAINT_TOLERANCE = 1e-6


Constraint = Bounds | LinearConstraint | NonlinearConstraint


def satisfies_constraint(
    control: np.ndarray,
    constraint: Constraint,
) -> bool:
    if isinstance(constraint, (Bounds, LinearConstraint)):
        lower_residual, upper_residual = constraint.residual(control)
        return bool(
            np.all(lower_residual >= -CONSTRAINT_TOLERANCE)
            and np.all(upper_residual >= -CONSTRAINT_TOLERANCE)
        )

    if isinstance(constraint, NonlinearConstraint):
        value = np.array(constraint.fun(control))
        lb = constraint.lb
        ub = constraint.ub
        return bool(
            np.all(value >= lb - CONSTRAINT_TOLERANCE)
            and np.all(value <= ub + CONSTRAINT_TOLERANCE)
        )

    raise TypeError(f"Unsupported constraint type: {type(constraint)!r}")


def satisfies_all_constraints(
    control: np.ndarray,
    constraints: list[Constraint],
) -> bool:
    return all(satisfies_constraint(control, constraint) for constraint in constraints)
