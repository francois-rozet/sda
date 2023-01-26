r"""Neural networks and modules."""

import torch
import torch.nn as nn
import torch.nn.functional as fun

from lampe.nn import MLP, ResMLP
from torch import Tensor, BoolTensor, Size
from torch.distributions import Normal
from typing import *
from zuko.utils import broadcast


class TimeEmbedding(nn.Module):
    def __init__(self, periods: int = 1):
        super().__init__()

        self.register_buffer('periods', torch.arange(periods) + 1)

    def forward(self, t: Tensor) -> Tensor:
        t = self.periods * torch.pi * t[..., None]
        t = torch.cat((t.cos(), t.sin()), dim=-1)

        return t


class ScoreNetwork(nn.Module):
    def __init__(
        self,
        features: int,
        periods: int = 1,
        residual: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.embedding = TimeEmbedding(periods)

        if residual:
            self.network = ResMLP(features + 2 * periods, features, **kwargs)
        else:
            self.network = MLP(features + 2 * periods, features, **kwargs)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        t = self.embedding(t)
        x, t = broadcast(x, t, ignore=1)

        return self.network(torch.cat((x, t), dim=-1))


class SequenceScoreNetwork(nn.Module):
    def __init__(self, features: int, window: int = 2, **kwargs):
        super().__init__()

        self.joint = ScoreNetwork(features * window, **kwargs)
        self.marginal = ScoreNetwork(features * (window - 1), **kwargs)

        self.window = window

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        x_j = x.unfold(-2, self.window, 1).flatten(-2)
        s_j = self.joint(x_j, t).unflatten(-1, (-1, self.window))
        s_j = self.fold(s_j)

        x_m = x[..., 1:-1, :].unfold(-2, self.window - 1, 1).flatten(-2)
        s_m = self.marginal(x_m, t).unflatten(-1, (-1, self.window - 1))
        s_m = self.fold(s_m)
        s_m = fun.pad(s_m, (0, 0, 1, 1))

        return s_j - s_m

    @staticmethod
    @torch.jit.script
    def fold(x: Tensor) -> Tensor:
        x = x.movedim(-3, -1)

        batch, (W, L) = x.shape[:-2], x.shape[-2:]
        kernel = torch.eye(W).flipud().to(x).unsqueeze(0)

        x = x.reshape(-1, W, L)
        x = fun.conv1d(x, kernel, padding=W - 1)
        x = x.reshape(batch + (-1,))

        return x.transpose(-2, -1)


class SubVariancePreservingSDE(nn.Module):
    def __init__(self, score: nn.Module, shape: Size):
        super().__init__()

        self.score = score
        self.dims = len(shape)

        self.register_buffer('zeros', torch.zeros(*shape))

    @staticmethod
    def alpha(t: Tensor) -> Tensor:
        return torch.exp(-8.0 * t**2)

    @staticmethod
    def beta(t: Tensor) -> Tensor:
        return 16.0 * t

    def forward(self, x: Tensor, t: Tensor) -> Tuple[Tensor, Tensor]:
        r"""Simulates the forward SDE until :math:`t`.

        Samples and returns

        .. math:: x(t) ~ N(x(t) | \sqrt{\alpha(t)} x, (1 - \alpha(t))^2 I)

        as well as the rescaled score

        .. math:: (1 - \alpha(t)) \nabla_{x(t)} \log p(x(t) | x) .
        """

        alpha = self.alpha(t).reshape(t.shape + (1,) * self.dims)

        z = torch.randn_like(x)
        x = alpha.sqrt() * x + (1 - alpha) * z

        return x, -z

    def reverse(
        self,
        shape: Size = (),
        steps: int = 64,
        corrections: int = 0,
        ratio: float = 1.0,
    ) -> Tensor:
        r"""Simulates the reverse SDE from :math:`t = 1` to :math:`0`.

        Arguments:
            shape: The batch shape.
            steps: The number of discrete time steps.
            corrections: The number of Langevin corrections per time steps.
            ratio: The signal-to-noise ratio of Langevin steps.
        """

        x = Normal(self.zeros, 1.0).sample(shape)
        time = torch.linspace(1.0, 0.0, steps + 1).to(x)

        with torch.no_grad():
            for t, dt in zip(time, time.diff()):
                alpha, beta = self.alpha(t), self.beta(t)

                # Corrector
                for _ in range(corrections):
                    z = torch.randn_like(x)
                    s = self.score(x, t)
                    eps = ratio * alpha * z.square().sum() / s.square().sum()

                    x = x + (1 - alpha) * eps * s + (1 - alpha) * (2 * eps).sqrt() * z

                # Predictor
                f = -beta * (x / 2 + (1 + alpha) * self.score(x, t))
                g = beta * (1 - alpha ** 2)

                z = torch.randn_like(x)
                x = x + f * dt + (g * dt).abs().sqrt() * z

        return x

    def loss(self, x: Tensor) -> Tensor:
        t = x.new_empty(x.shape[:-self.dims]).uniform_()
        x, target = self.forward(x, t)

        return (self.score(x, t) - target).square().mean()


class ComposedScore(nn.Module):
    def __init__(self, *scores: nn.Module):
        super().__init__()

        self.scores = nn.ModuleList(scores)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        return sum(s(x, t) for s in self.scores)


class ImputationScore(nn.Module):
    def __init__(
        self,
        y: Tensor,
        mask: BoolTensor,
        sigma: Union[float, Tensor] = 1e-3,
    ):
        super().__init__()

        self.register_buffer('y', y)
        self.register_buffer('mask', mask)
        self.register_buffer('sigma', torch.as_tensor(sigma))

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        alpha = SubVariancePreservingSDE.alpha(t)

        s = -(x - alpha.sqrt() * self.y) / (alpha * self.sigma ** 2 + (1 - alpha) ** 2)
        s = self.mask * s
        s = (1 - alpha) * s

        return s
