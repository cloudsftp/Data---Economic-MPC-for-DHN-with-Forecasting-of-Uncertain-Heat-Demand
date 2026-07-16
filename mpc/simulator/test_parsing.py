import pandas as pd
import numpy as np

from .parsing import (
    _compute_objective,
    _compute_gradients,
    _compute_min_temperature,
    _compute_min_temperature_gradients,
)


def test_objective():
    data = pd.DataFrame(
        {
            "S0.0.Q": [1.0, 2.0, 3.0, 4.0],
            "S0.1.Q": [-11.0, -10.0, -9.0, -8.0],
        }
    )

    result = _compute_objective(
        data,
        ["S0.0.Q", "S0.1.Q"],
        [],
    )  # TODO: also incorporate consumer Q

    assert result == 5.25


def test_objective_gradients():
    data = pd.DataFrame(
        {
            "S0.0.Q|S1": [2.0, 2.0, 2.0],
            "S0.0.Q|S2": [1.0, 2.0, 3.0],
            "S0.1.Q|S1": [-1.0, -1.0, -1.0],
            "S0.1.Q|S2": [-3.0, -2.0, -1.0],
        }
    )

    result = _compute_gradients(
        data,
        ["S0.0.Q", "S0.1.Q"],
        [],
        ["S1", "S2"],
    )  # TODO: also incorporate consumer Q

    assert np.array_equal(result, [-0.5, 0])


def test_min_temperature():
    data = pd.DataFrame(
        {
            "C0.0.e": [1.0, 2.0, 1.0],
            "C1.0.e": [3.0, 2.0, 0.0],
        }
    )

    result = _compute_min_temperature(data, ["C0.0.e", "C1.0.e"], 0, 0)

    assert np.array_equal(
        result,
        [1.0, 2.0, 1.0, 3.0, 2.0, 0.0],
    )


def test_min_temperature_gradients():
    data = pd.DataFrame(
        {
            "C0.0.e|S1": [0.0, 1.0, 2.0],
            "C0.0.e|S2": [1.0, 2.0, 3.0],
            "C1.0.e|S1": [0.5, 0.0, -1.0],
            "C1.0.e|S2": [0.0, 1.0, 2.0],
        }
    )

    result = _compute_min_temperature_gradients(data, ["C0.0.e", "C1.0.e"], ["S2"], 0)

    assert np.array_equal(
        result,
        [[1.0], [2.0], [3.0], [0.0], [1.0], [2.0]],
    )
