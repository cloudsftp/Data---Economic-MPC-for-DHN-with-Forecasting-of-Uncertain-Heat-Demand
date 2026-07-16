import numpy as np

from mpc.control import _changes_in_temperature_matrix


def test_changes_in_temperature_matrix():
    n = 3

    matrix = _changes_in_temperature_matrix(n)

    assert np.array_equal(
        matrix,
        [
            [-1.0, 1.0, 0.0],
            [0.0, -1.0, 1.0],
        ],
    )
