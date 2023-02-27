r"""Score modules"""

import torch
import torch.nn as nn

from torch import Tensor, BoolTensor, Size
from torch.distributions import Normal
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
        features: The number of features (last dimension).
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


class SubVariancePreservingSDE(nn.Module):
    r"""Creates a sub-variance preserving SDE.

    Arguments:
        score: A score estimator.
        shape: The shape of the state space.
    """

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

    def sample(
        self,
        shape: Size = (),
        steps: int = 64,
        corrections: int = 0,
        ratio: float = 1.0,
    ) -> Tensor:
        r"""Samples from the reverse SDE.

        Arguments:
            shape: The batch shape.
            steps: The number of discrete time steps.
            corrections: The number of Langevin corrections per time steps.
            ratio: The signal-to-noise ratio of Langevin steps.
        """

        x = Normal(self.zeros, 1.0).sample(shape)
        time = torch.linspace(1.0, 0.0, steps + 1).square().to(x)

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
                f = -beta / 2 * (x + (1 + alpha) * self.score(x, t))
                x = x + f * dt

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
