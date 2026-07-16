from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from math import sqrt

from serde.json import from_json

from mpc.performance import absolute_performance_index
from mpc.mpc import MPCResult
from mpc.simulator import Simulator


@dataclass
class HorizonMPCResult:
    horizon: int
    result: MPCResult

    def optimal_control_distance(self, other_control: list[float]) -> float:
        return _distance(self.result.optimal_control, other_control)

    def first_control_distance(self, other_control: list[float]) -> float:
        return abs(self.result.optimal_control[1] - other_control[1])

    def absolute_performance_index(
        self,
        simulator: Simulator,
        other_control: list[float],
    ) -> float:
        return absolute_performance_index(
            simulator,
            self.result.optimal_control,
            other_control,
        )

    def median_time(self) -> float:
        times = self.result.times

        times.sort()
        return times[int((len(times) / 2))]

    def average_time(self) -> float:
        times = self.result.times

        return sum(times) / len(times)


DIR_SUFFIX = "-hour"


def load_mpc_results(
    path: Path = Path("data/results/actual-no-buffer-8-steps"),
) -> list[HorizonMPCResult]:
    results = []

    for path in path.iterdir():
        if not path.is_dir() or DIR_SUFFIX not in path.name:
            continue

        horizon_string = path.name.removesuffix(DIR_SUFFIX)
        horizon = int(horizon_string)

        path = path / "optimal_control.json"

        with open(path) as f:
            content = f.read()
            mpc_result = from_json(MPCResult, content)

            results.append(HorizonMPCResult(horizon, mpc_result))

    results.sort(key=lambda result: result.horizon)
    return results


def _distance(
    a: list[float],
    b: list[float],
) -> float:
    assert len(a) == len(
        b
    ), f"lengths of control must be equal, got len(a)={len(a)} and len(b)={len(b)}"

    distances = [a - b for a, b in zip(a, b)]

    distance = 0.0

    for i in range(len(a) - 1):
        distance += (
            distances[i] ** 2 + distances[i] * distances[i + 1] + distances[i + 1] ** 2
        )

    return sqrt(distance / 3)
