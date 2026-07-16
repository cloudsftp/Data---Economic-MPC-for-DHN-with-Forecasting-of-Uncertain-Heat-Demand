from dataclasses import dataclass
from typing import Callable, Protocol

from loguru import logger
import numpy as np

from mpc.simulator import Simulator

ALPHA = 1.0e1
BETA = 1.0e0
GAMMA = 1.0e-6


class ObjectiveFunction(Protocol):
    def build_objective(
        self,
        simulator: Simulator,
    ) -> Callable[[np.ndarray], float]: ...
    def build_jacobian(
        self,
        simulator: Simulator,
    ) -> Callable[[np.ndarray], np.ndarray]: ...


@dataclass
class J1(ObjectiveFunction):
    alpha: float = ALPHA
    beta: float = BETA
    dt: float = 1.0

    def ell(self, u_i: float, u_l: float) -> float:
        return self.dt * (
            self.alpha * (((u_i - u_l) / self.dt) ** 2)
            + self.beta * (1.0 / 3.0 * (u_i - u_l) ** 2 + u_l * (u_i - u_l) + u_l**2)
        )

    def _objective(
        self,
        parameters: list[float],
    ) -> float:
        sum_ = 0.0
        for i in range(1, len(parameters)):
            u_i = parameters[i]
            u_l = parameters[i - 1]

            sum_ += self.ell(u_i, u_l)
        return sum_

    def build_objective(
        self,
        simulator: Simulator,
    ) -> Callable[[np.ndarray], float]:
        def objective(parameters: np.ndarray) -> float:
            return self._objective([simulator.fixed_control[-1]] + parameters.tolist())

        return objective

    def build_jacobian(
        self,
        simulator: Simulator,
    ) -> Callable[[np.ndarray], np.ndarray]:
        def jacobian(parameters: np.ndarray) -> np.ndarray:
            gradients = []
            for j in range(len(parameters)):
                u_j = parameters[j]
                if j > 0:
                    u_l = parameters[j - 1]
                else:
                    u_l = simulator.fixed_control[-1]

                gradient = 2.0 * self.alpha * (u_j - u_l) / self.dt + self.beta * (
                    2.0 / 3.0 * (u_j - u_l) + u_l
                )

                if j < len(parameters) - 1:
                    u_r = parameters[j + 1]
                    gradient += -2.0 * self.alpha * (
                        u_r - u_j
                    ) / self.dt + self.beta * (
                        -2.0 / 3.0 * (u_r - u_j) + (u_r - u_j) - u_j + 2.0 * u_j
                    )

                gradient *= self.dt

                gradients.append(gradient)

            logger.debug(f"gradients for {parameters}: {gradients}")

            return np.array(gradients).transpose()

        return jacobian


@dataclass
class J2(ObjectiveFunction):
    alpha: float = ALPHA
    beta: float = BETA
    gamma: float = GAMMA
    dt: float = 1.0

    def build_objective(
        self,
        simulator: Simulator,
    ) -> Callable[[np.ndarray], float]:
        j1 = J1(self.alpha, self.beta, self.dt).build_objective(simulator)
        violation = simulator.min_temperature_violation()

        def objective(parameters: np.ndarray) -> float:
            j = j1(parameters)
            v = self.gamma * violation(parameters)
            logger.debug(f"objective: {j}, violation: {v}")

            return j + v

        return objective

    def build_jacobian(
        self,
        simulator: Simulator,
    ) -> Callable[[np.ndarray], np.ndarray]:
        return simulator.jacobian()


@dataclass
class J3(ObjectiveFunction):
    def build_objective(
        self,
        simulator: Simulator,
    ) -> Callable[[np.ndarray], float]:
        return simulator.objective()

    def build_jacobian(
        self,
        simulator: Simulator,
    ) -> Callable[[np.ndarray], np.ndarray]:
        return simulator.jacobian()
