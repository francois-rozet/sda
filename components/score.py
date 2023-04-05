r"""Score modules"""

import math
import torch
import torch.nn as nn

from torch import Size, Tensor, BoolTensor
from typing import *
from zuko.utils import broadcast

from .nn import *


class TimeEmbedding(nn.Sequential):
    r"""Creates a time embedding.

    Arguments:
        features: The number of embedding features.
    """

    def __init__(self, features: int):
        super().__init__(
            nn.Linear(128, 256),
            nn.SiLU(),
            nn.Linear(256, features),
        )

        self.register_buffer('freqs', torch.pi / 2 * 1e3 ** torch.linspace(0, 1, 64))

    def forward(self, t: Tensor) -> Tensor:
        t = self.freqs * t.unsqueeze(dim=-1)
        t = torch.cat((t.cos(), t.sin()), dim=-1)

        return super().forward(t)


class ScoreNet(nn.Module):
    r"""Creates a score network.

    Arguments:
        features: The number of features.
        embedding: The number of time embedding features.
    """

    def __init__(self, features: int, embedding: int = 16, **kwargs):
        super().__init__()

        self.embedding = TimeEmbedding(embedding)
        self.network = MLP(features + embedding, features, **kwargs)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        t = self.embedding(t)
        x, t = broadcast(x, t, ignore=1)
        x = torch.cat((x, t), dim=-1)

        return self.network(x)


class ScoreUNet(nn.Module):
    r"""Creates a U-Net score network.

    Arguments:
        channels: The number of channels.
        embedding: The number of time embedding features.
    """

    def __init__(self, channels: int, embedding: int = 64, **kwargs):
        super().__init__()

        self.embedding = TimeEmbedding(embedding)
        self.network = UNet(channels, channels, embedding, **kwargs)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        dims = self.network.spatial + 1

        y = x.reshape(-1, *x.shape[-dims:])
        t = t.reshape(-1)
        t = self.embedding(t)

        return self.network(y, t).reshape(x.shape)


class MCScoreNet(nn.Module):
    r"""Creates a score network for a Markov chain.

    Arguments:
        features: The number of features.
        order: The order of the Markov chain.
    """

    def __init__(self, features: int, order: int = 1, **kwargs):
        super().__init__()

        self.order = order

        if kwargs.get('spatial', 0) > 0:
            build = ScoreUNet
        else:
            build = ScoreNet

        self.kernel = build(features * (2 * order + 1), **kwargs)

    def forward(
        self,
        x: Tensor,  # (B, L, C, H, W)
        t: Tensor,  # ()
    ) -> Tensor:
        x = self.unfold(x, self.order)
        s = self.kernel(x, t)
        s = self.fold(s, self.order)

        return s

    @staticmethod
    @torch.jit.script_if_tracing
    def unfold(x: Tensor, order: int) -> Tensor:
        x = x.unfold(1, 2 * order + 1, 1)
        x = x.movedim(-1, 2)
        x = x.flatten(2, 3)

        return x

    @staticmethod
    @torch.jit.script_if_tracing
    def fold(x: Tensor, order: int) -> Tensor:
        x = x.unflatten(2, (2 * order  + 1, -1))

        return torch.cat((
            x[:, 0, :order],
            x[:, :, order],
            x[:, -1, -order:],
        ), dim=1)


class CompositeMCScoreNet(nn.Module):
    r"""Creates a composite score network for a Markov chain.

    Arguments:
        features: The number of features.
        order: The order of the Markov chain.
    """

    def __init__(self, features: int, order: int = 1, **kwargs):
        super().__init__()

        self.order = order

        if kwargs.get('spatial', 0) > 0:
            build = ScoreUNet
        else:
            build = ScoreNet

        self.kernel_j = build(features * (order + 1), **kwargs)
        self.kernel_m = build(features * order, **kwargs)

    def forward(
        self,
        x: Tensor,  # (B, L, C, H, W)
        t: Tensor,  # ()
    ) -> Tensor:
        x_j = self.unfold(x, self.order + 1)
        s_j = self.kernel_j(x_j, t)
        s_j = self.fold(s_j, self.order + 1)

        x_m = x[:, 1:-1]
        x_m = self.unfold(x_m, self.order)
        s_m = self.kernel_m(x_m, t)
        s_m = self.fold(s_m, self.order)
        s_m = self.pad(s_m, (1, 1))

        return s_j - s_m

    @staticmethod
    @torch.jit.script_if_tracing
    def unfold(x: Tensor, window: int) -> Tensor:
        if window == 1:
            return x

        x = x.unfold(1, window, 1)
        x = x.movedim(-1, 2)
        x = x.flatten(2, 3)

        return x

    @staticmethod
    @torch.jit.script_if_tracing
    def fold(x: Tensor, window: int) -> Tensor:
        if window == 1:
            return x

        kernel = torch.eye(window, dtype=x.dtype, device=x.device)
        kernel = kernel.flipud().unsqueeze(0)

        x = x.unflatten(2, (window, -1))
        x = x.movedim((1, 2), (-1, -2))

        shape = x.shape[:-2]

        x = x.flatten(0, -3)
        x = torch.nn.functional.conv1d(x, kernel, padding=window - 1)
        x = x.reshape(shape + (-1,))
        x = x.movedim(-1, 1)

        return x

    @staticmethod
    @torch.jit.script_if_tracing
    def pad(x: Tensor, padding: Tuple[int, int]) -> Tensor:
        return torch.nn.functional.pad(x, (0, 0) * (x.dim() - 2) + padding)


class VPSDE(nn.Module):
    r"""Creates a noise scheduler for the variance preserving (VP) SDE.

    .. math::
        \mu(t) & = \alpha(t) \\
        \sigma(t)^2 & = 1 - \alpha(t)^2 + \epsilon^2

    Arguments:
        score: A score estimator :math:`s_\phi(x, t)`.
        shape: The event shape.
        epsilon: A numerical stability term.
    """

    def __init__(self, score: nn.Module, shape: Size, epsilon: float = 1e-3):
        super().__init__()

        self.score = score
        self.shape = shape
        self.epsilon = epsilon

        self.register_buffer('ones', torch.ones(shape))

    def alpha(self, t: Tensor) -> Tensor:
        return torch.exp(math.log(self.epsilon) * t**2)

    def mu(self, t: Tensor) -> Tensor:
        return self.alpha(t)

    def sigma(self, t: Tensor) -> Tensor:
        return (1 - self.alpha(t) ** 2 + self.epsilon ** 2).sqrt()

    def forward(self, x: Tensor, t: Tensor, train: bool = False) -> Tensor:
        r"""Samples from the perturbation kernel :math:`p(x(t) | x)`."""

        t = t.reshape(t.shape + (1,) * len(self.shape))

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

        x = torch.normal(0, self.ones.expand(shape + self.shape))
        x = x.reshape(-1, *self.shape)

        time = torch.linspace(1, 0, steps + 1).square().to(x)

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

        return x.reshape(shape + self.shape)

    def loss(self, x: Tensor) -> Tensor:
        r"""Returns the rescaled denoising score matching loss."""

        t = torch.rand(x.shape[0], dtype=x.dtype, device=x.device)
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


class ImputationScore(nn.Module):
    r"""Creates a score module for imputation problems."""

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


class LinearGaussianScore(nn.Module):
    r"""Creates a score module for linear Gaussian inverse problems."""

    def __init__(
        self,
        y: Tensor,
        f: Callable[[Tensor], Tensor],  # A @ x
        diag: Union[float, Tensor],  # diag(A @ A^T)
        noise: Union[float, Tensor],
        sde: VPSDE,
    ):
        super().__init__()

        self.register_buffer('y', y)
        self.register_buffer('diag', torch.as_tensor(diag))
        self.register_buffer('noise', torch.as_tensor(noise))

        self.f = f
        self.score = sde.score
        self.mu = sde.mu
        self.sigma = sde.sigma

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        with torch.enable_grad():
            x = x.detach().requires_grad_(True)

            err = self.f(x) - self.mu(t) * self.y
            var = (self.mu(t) * self.noise) ** 2 + self.sigma(t) ** 2 * self.diag
            var = var / self.sigma(t)

            log_p = -(err ** 2 / var).sum() / 2

        score, = torch.autograd.grad(log_p, x)

        return self.score(x, t) + score
