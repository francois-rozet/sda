r"""Score modules"""

import math
import torch
import torch.nn as nn

from torch import Size, Tensor, BoolTensor
from typing import *
from zuko.utils import broadcast

from .nn import MLP, FCNN


class TimeEmbedding(nn.Module):
    r"""Creates a time embedding.

    Arguments:
        freqs: The number of embedding frequencies.
    """

    def __init__(self, freqs: int = 3):
        super().__init__()

        self.register_buffer('freqs', torch.arange(1, freqs + 1) * torch.pi)

    def forward(self, t: Tensor) -> Tensor:
        t = self.freqs * t[..., None]
        t = torch.cat((t.cos(), t.sin()), dim=-1)

        return t


class ScoreNet(nn.Module):
    r"""Creates a score network.

    Arguments:
        features: The number of features.
        freqs: The number of embedding frequencies.
        kwargs: The keyword arguments passed to :class:`MLP` or :class:`FCNN`.
    """

    def __init__(
        self,
        features: int,
        freqs: int = 3,
        spatial: int = 0,
        **kwargs,
    ):
        super().__init__()

        self.embedding = TimeEmbedding(freqs)

        if spatial > 0:
            self.network = FCNN(features + 2 * freqs, features, spatial=spatial, **kwargs)
        else:
            self.network = MLP(features + 2 * freqs, features, **kwargs)

        self.spatial = spatial

    def forward(
        self,
        x: Tensor,  # (*, H, W, C)
        t: Tensor,  # (*,)
    ) -> Tensor:
        t = self.embedding(t)
        t = t.unflatten(-1, (1,) * self.spatial + (-1,))

        x, t = broadcast(x, t, ignore=1)
        x = torch.cat((x, t), dim=-1)
        x = x.movedim(-1, -(self.spatial + 1))
        x = self.network(x)
        x = x.movedim(-(self.spatial + 1), -1)

        return x


class MarkovChainScoreNet(nn.Module):
    r"""Creates a score network for a Markov chain.

    Arguments:
        features: The number of features.
        spatial: The number of spatial dimensions.
        order: The order of the Markov chain.
    """

    def __init__(self, features: int, spatial: int = 0, order: int = 1, **kwargs):
        super().__init__()

        self.joint = ScoreNet(features * (order + 1), spatial=spatial, **kwargs)
        self.marginal = ScoreNet(features * order, spatial=spatial, **kwargs)

        self.spatial = spatial
        self.order = order

    def forward(
        self,
        x: Tensor,  # (*, L, H, W, C)
        t: Tensor,  # (*,)
    ) -> Tensor:
        t = t.unsqueeze(-1)
        dim = -(self.spatial + 2)

        x_j = x.unfold(dim, self.order + 1, 1).flatten(-2)
        s_j = self.joint(x_j, t).unflatten(-1, (-1, self.order + 1))
        s_j = self.fold(s_j, dim=dim - 1)

        x_m = x.narrow(dim, start=1, length=x.shape[dim] - 2)
        x_m = x_m.unfold(dim, self.order, 1).flatten(-2)
        s_m = self.marginal(x_m, t).unflatten(-1, (-1, self.order))
        s_m = self.fold(s_m, dim=dim - 1)
        s_m = self.pad(s_m, (1, 1), dim=dim)

        return s_j - s_m

    @staticmethod
    @torch.jit.script_if_tracing
    def fold(x: Tensor, dim: int) -> Tensor:
        dim = dim % x.dim()
        x = x.movedim(dim, -1)

        batch, (C, L) = x.shape[:-2], x.shape[-2:]
        kernel = torch.eye(C).flipud().to(x).unsqueeze(0)

        x = x.reshape(-1, C, L)
        x = torch.nn.functional.conv1d(x, kernel, padding=C - 1)
        x = x.reshape(batch + (-1,))

        return x.movedim(-1, dim)

    @staticmethod
    @torch.jit.script_if_tracing
    def pad(x: Tensor, padding: Tuple[int, int], dim: int) -> Tensor:
        return torch.nn.functional.pad(x, (0, 0) * (x.dim() - dim % x.dim() - 1) + padding)


class VPSDE(nn.Module):
    r"""Creates a noise scheduler for the variance preserving (VP) SDE.

    The goal of denoising score matching (DSM) is to train a score estimator
    :math:`s_\phi(x, t)` at approximating the rescaled score

    .. math:: s(x(t), t) = \sigma(t) \nabla_{x(t)} \log p(x(t))

    by minimizing the rescaled score matching objective

    .. math:: \arg \min_\phi \mathbb{E}_{p(x) p(t) p(x(t) | x)} \Big[ \big\|
        s_\phi(x(t), t) - \sigma(t) \nabla_{x(t)} \log p(x(t) | x) \big\|_2^2 \Big]

    where :math:`p(x)` is an unknown distribution and :math:`p(x(t) | x)` is a
    perturbation kernel of the form

    .. math:: p(x(t) | x) = \mathcal{N}(x(t) | \mu(t) x, \sigma(t)^2 I) .

    In the case of the variance preserving (VP) SDE,

    .. math::
        \mu(t) & = \alpha(t) \\
        \sigma(t)^2 & = 1 - \alpha(t)^2 + \epsilon^2

    with :math:`\alpha(t) = \exp(\log(\epsilon) t^2)`. After training, we can sample
    from :math:`p(x(0))` by first sampling :math:`x(1) \sim p(x(1)) \approx
    \mathbb{N}(0, 1)` and then denoising it iteratively with

    .. math:: x(t - \Delta t) \approx \frac{\mu(t - \Delta t)}{\mu(t)} x(t) +
        (\frac{\mu(t - \Delta t)}{\mu(t)} \sigma(t) - \sigma(t - \Delta)) s(x(t), t)

    Arguments:
        score: A score estimator :math:`s_\phi(x, t)`.
        shape: The event shape.
        epsilon: A numerical stability term.
    """

    def __init__(self, score: nn.Module, shape: Size, epsilon: float = 1e-3):
        super().__init__()

        self.score = score
        self.shape = shape
        self.dims = len(shape)
        self.epsilon = epsilon

        self.register_buffer('ones', torch.ones(shape))

    def alpha(self, t: Tensor) -> Tensor:
        return torch.exp(math.log(self.epsilon) * t**2)

    def mu(self, t: Tensor) -> Tensor:
        return self.alpha(t)

    def sigma(self, t: Tensor) -> Tensor:
        return (1 - self.alpha(t) ** 2 + self.epsilon ** 2).sqrt()

    def forward(
        self,
        x: Tensor,
        t: Tensor,
        train: bool = False,
    ) -> Tensor:
        r"""Samples from the perturbation kernel :math:`p(x(t) | x)`."""

        t = t.reshape(t.shape + (1,) * self.dims)

        z = torch.randn_like(x)
        x = self.mu(t) * x + self.sigma(t) * z

        if train:
            return x, -z
        else:
            return x

    def sample(
        self,
        shape: Size = (),
        steps: int = 64,
        corrections: int = 0,
        amplitude: float = 1.0,
    ) -> Tensor:
        r"""Samples from :math:`p(x(0))`.

        Arguments:
            shape: The batch shape.
            steps: The number of discrete time steps.
            corrections: The number of Langevin corrections per time steps.
            amplitude: The amplitude of Langevin steps.
        """

        x = torch.normal(0.0, self.ones.expand(shape + self.shape))
        time = torch.linspace(1.0, 0.0, steps + 1).square().to(x)

        with torch.no_grad():
            for t, dt in zip(time, time.diff()):
                # Predictor
                r = self.mu(t + dt) / self.mu(t)
                x = r * x + (r * self.sigma(t) - self.sigma(t + dt)) * self.score(x, t)

                # Corrector
                for _ in range(corrections):
                    z = torch.randn_like(x)
                    s = self.score(x, t)
                    eps = amplitude * z.square().sum() / s.square().sum()

                    x = x + self.sigma(t + dt) * (eps * s + (2 * eps).sqrt() * z)

        return x

    def loss(self, x: Tensor) -> Tensor:
        t = x.new_empty(x.shape[:-self.dims]).uniform_()
        x, target = self.forward(x, t, train=True)

        return (self.score(x, t) - target).square().mean()


class SubVPSDE(VPSDE):
    r"""Creates a noise scheduler for the sub-variance preserving (sub-VP) SDE.

    .. math::
        \mu(t) & = \alpha(t) \\
        \sigma(t)^2 & = (1 - \alpha(t)^2 + \epsilon)^2
    """

    def sigma(self, t: Tensor) -> Tensor:
        return 1 - self.alpha(t) ** 2 + self.epsilon


class SubSubVPSDE(VPSDE):
    r"""Creates a noise scheduler for the sub-sub-VP SDE.

    .. math::
        \mu(t) & = \alpha(t) \\
        \sigma(t)^2 & = (1 - \alpha(t) + \epsilon)^2
    """

    def sigma(self, t: Tensor) -> Tensor:
        return 1 - self.alpha(t) + self.epsilon


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
        sde: VPSDE,
    ):
        super().__init__()

        self.register_buffer('y', y)
        self.register_buffer('mask', mask)

        self.score = sde.score
        self.mu = sde.mu
        self.sigma = sde.sigma

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        return torch.where(
            self.mask,
            (self.mu(t) * self.y - x) / self.sigma(t),
            self.score(x, t),
        )
