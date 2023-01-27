r"""Markov chains."""

import abc
import torch

from torch import Tensor, Size
from torch.distributions import MultivariateNormal
from typing import *


class MarkovChain(abc.ABC):
    r"""Abstract Markov chain class

    Wikipedia:
        https://wikipedia.org/wiki/Markov_chain
    """

    @abc.abstractmethod
    def prior(self, shape: Size = ()) -> Tensor:
        r""" x_0 ~ p(x_0) """

        pass

    @abc.abstractmethod
    def transition(self, x: Tensor) -> Tensor:
        r""" x_i ~ p(x_i | x_{i-1}) """

        pass

    def trajectory(self, x: Tensor, steps: int, last: bool = False) -> Tensor:
        r""" (x_0, x_1, ..., x_n) ~ p(x_0) \prod_i p(x_i | x_{i-1}) """

        if last:
            for _ in range(steps):
                x = self.transition(x)

            return x
        else:
            X = [x]

            for _ in range(steps):
                x = self.transition(x)
                X.append(x)

            return torch.stack(X)


class LinearGaussian(MarkovChain):
    r"""Linear-Gaussian Markov chain"""

    def __init__(
        self,
        mu_0: Tensor,
        Sigma_0: Tensor,
        A: Tensor,
        b: Tensor,
        Sigma_x: Tensor,
    ):
        super().__init__()

        self.mu_0, self.Sigma_0 = mu_0, Sigma_0
        self.A, self.b, self.Sigma_x = A, b, Sigma_x

    def prior(self, shape: Size = ()) -> Tensor:
        r""" x_0 ~ N(\mu_0, \Sigma_0) """

        return MultivariateNormal(self.mu_0, self.Sigma_0).sample(shape)

    def transition(self, x: Tensor) -> Tensor:
        r""" x_i ~ N(A x_{i-1} + b, \Sigma_x) """

        return MultivariateNormal(x @ self.A.T, self.Sigma_x).sample()


class DampedSpring(LinearGaussian):
    r"""Damped Spring hidden Markov model

    This class describes the (linearized) dynamics of a mass attached to a spring,
    subject to wind and friction.
    """

    def __init__(self, dt: float = 0.1):
        # Prior
        mu_0 = torch.tensor([1.0, 0.0, 0.0, 0.0])
        Sigma_0 = torch.tensor([1.0, 1.0, 1.0, 1.0]).diag()

        # Transition
        A = torch.tensor([
            [1.0, dt, dt**2 / 2, 0.0],
            [0.0, 1.0, dt, 0.0],
            [-0.5, -0.1, 0.0, 0.2],
            [0.0, 0.0, 0.0, 0.99],
        ])
        b = torch.zeros(4)
        Sigma_x = torch.tensor([0.1, 0.1, 0.1, 1.0]).diag() * dt

        super().__init__(mu_0, Sigma_0, A, b, Sigma_x)


class DifferentialGaussian(MarkovChain):
    r"""Differential-Gaussian Markov chain"""

    def __init__(self, wiener: float = 1.0, dt: float = 0.01, steps: int = 1):
        super().__init__()

        self.wiener = wiener
        self.dt, self.steps = dt, steps

    @staticmethod
    def rk4(f: Callable[[Tensor], Tensor], x: Tensor, dt: float) -> Tensor:
        r""""Performs a step of the fourth-order Runge-Kutta integration scheme.

        Wikipedia:
            https://wikipedia.org/wiki/Runge-Kutta_methods
        """

        k1 = f(x)
        k2 = f(x + dt * k1 / 2)
        k3 = f(x + dt * k2 / 2)
        k4 = f(x + dt * k3)

        return x + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6

    @abc.abstractmethod
    def f(self, x: Tensor) -> Tensor:
        r""" f(x) = \frac{dx}{dt} """

        pass

    def transition(self, x: Tensor) -> Tensor:
        for _ in range(self.steps):
            x = self.rk4(self.f, x, self.dt / self.steps)

        return torch.normal(x, self.wiener * self.dt**0.5)


class Lorenz63(DifferentialGaussian):
    r"""Lorenz 1963 as a Markov chain

    Wikipedia:
        https://wikipedia.org/wiki/Lorenz_system
    """

    def __init__(
        self,
        sigma: float = 10.0,
        rho: float = 28.0,
        beta: float = 8 / 3,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.sigma, self.rho, self.beta = sigma, rho, beta

    def prior(self, shape: Size = ()) -> Tensor:
        modes = torch.tensor([
            [self.sigma, self.sigma, self.rho],
            [-self.sigma, -self.sigma, self.rho],
        ])

        i = torch.randint(2, shape)

        return torch.normal(modes[i], self.sigma / 4)

    def f(self, x: Tensor) -> Tensor:
        return torch.stack((
            self.sigma * (x[..., 1] - x[..., 0]),
            x[..., 0] * (self.rho - x[..., 2]) - x[..., 1],
            x[..., 0] * x[..., 1] - self.beta * x[..., 2],
        ), dim=-1)


class Lorenz96(DifferentialGaussian):
    r"""Lorenz 1996 as a Markov chain

    Wikipedia:
        https://wikipedia.org/wiki/Lorenz_96_model
    """

    def __init__(
        self,
        n: int = 32,
        F: float = 16.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.n, self.F = n, F

    def prior(self, shape: Size = ()) -> Tensor:
        return torch.randn(*shape, self.n)

    def f(self, x: Tensor) -> Tensor:
        x1, x2, x3 = [torch.roll(x, i, dims=-1) for i in (1, -2, -1)]

        return (x1 - x2) * x3 - x + self.F


class LotkaVolterra(DifferentialGaussian):
    r"""Lotka-Volterra as a Markov chain

    Wikipedia:
        https://wikipedia.org/wiki/Lotka-Volterra_equations
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 1.0,
        delta: float = 1.0,
        gamma: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.alpha, self.beta = alpha, beta
        self.delta, self.gamma = delta, gamma

    def prior(self, shape: Size = ()) -> Tensor:
        return torch.rand(*shape, 2)

    def f(self, x: Tensor) -> Tensor:
        return torch.stack((
            self.alpha - self.beta * x[..., 1].exp(),
            self.delta * x[..., 0].exp() - self.gamma,
        ), dim=-1)
