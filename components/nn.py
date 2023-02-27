r"""Neural networks and modules"""

import torch
import torch.nn as nn

from torch import Tensor
from typing import *
from zuko.nn import LayerNorm, MLP


class Residual(nn.Sequential):
    r"""Creates a residual block."""

    def forward(self, x: Tensor) -> Tensor:
        return x + super().forward(x)


class FCNN(nn.Sequential):
    r"""Creates a fully convolutional neural network (FCNN).

    The architecture is inspired by ConvNeXt blocks which mix depthwise and 1 by 1
    convolutions to improve the efficiency/accuracy trade-off.

    References:
        | A ConvNet for the 2020s (Lui et al., 2022)
        | https://arxiv.org/abs/2201.03545

    Arguments:
        in_channels: The number of input channels.
        out_channels: The number of output channels.
        hidden_channels: The number of hidden channels.
        hidden_blocks: The number of hidden blocks. Each block consists in an optional
            normalization, a depthwise convolution, an activation and a 1 by 1
            convolution.
        kernel_size: The size of the convolution kernels.
        activation: The activation function constructor.
        normalize: Whether channels are normalized or not.
        spatial: The number of spatial dimensions. Can be either 1, 2 or 3.
        kwargs: Keyword arguments passed to :class:`torch.nn.Conv2d`.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int = 64,
        hidden_blocks: int = 3,
        kernel_size: int = 5,
        activation: Callable[[], nn.Module] = nn.ReLU,
        normalize: bool = False,
        spatial: int = 2,
        **kwargs,
    ):
        convolution = {
            1: nn.Conv1d,
            2: nn.Conv2d,
            3: nn.Conv3d,
        }.get(spatial)

        if normalize:
            normalization = lambda: LayerNorm(dim=-(spatial + 1))
        else:
            normalization = nn.Identity

        layers = [
            convolution(
                in_channels,
                hidden_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                **kwargs,
            )
        ]

        for i in range(hidden_blocks):
            layers.append(
                Residual(
                    normalization(),
                    convolution(
                        hidden_channels,
                        hidden_channels * 2,
                        groups=hidden_channels,
                        kernel_size=kernel_size,
                        padding=kernel_size // 2,
                        **kwargs,
                    ),
                    activation(),
                    convolution(hidden_channels * 2, hidden_channels, kernel_size=1),
                )
            )

        layers.append(
            convolution(
                hidden_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                **kwargs,
            )
        )

        super().__init__(*layers)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.spatial = spatial

    def forward(self, x: Tensor) -> Tensor:
        dim = -(self.spatial + 1)

        y = x.reshape(-1, *x.shape[dim:])
        y = super().forward(y)
        y = y.reshape(*x.shape[:dim], *y.shape[dim:])

        return y


class S4DLayer(nn.Module):
    r"""Creates a structured state space sequence diagonal (S4D) layer.

    References:
        | On the Parameterization and Initialization of Diagonal State Space Models (Gu et al., 2022)
        | https://arxiv.org/abs/2206.11893

    Arguments:
        channels: The number of channels.
        space: The size of the state space.
    """

    def __init__(self, channels: int, space: int = 16):
        super().__init__()

        self.log_dt = nn.Parameter(torch.empty(channels).uniform_(-7.0, -3.0))
        self.a_real = nn.Parameter(torch.full((channels, space), 0.5).log())
        self.a_imag = nn.Parameter(torch.pi * torch.arange(space).expand(channels, -1))
        self.c = nn.Parameter(torch.randn(channels, space, 2))

    def extra_repr(self) -> str:
        channels, space = self.a_real.shape
        return f'{channels}, space={space}'

    def kernel(self, length: int) -> Tensor:
        dt = self.log_dt.exp()
        a = torch.complex(-self.a_real.exp(), self.a_imag)
        c = torch.view_as_complex(self.c)

        a_dt = a * dt[..., None]
        b_c = c * (a_dt.exp() - 1) / a

        power = torch.arange(length, device=a.device)
        vandermonde = (a_dt[..., None] * power).exp()

        return 2 * torch.einsum('...i,...ij', b_c, vandermonde).real

    def forward(self, x: Tensor) -> Tensor:
        length = x.shape[-1]
        k = self.kernel(length)

        k = torch.fft.rfft(k, n=2 * length)
        x = torch.fft.rfft(x, n=2 * length)
        y = torch.fft.irfft(k * x)[..., :length]

        return y


class S4DBlock(nn.Module):
    r"""Creates a S4D bidirectional block.

    Arguments:
        channels: The number of channels.
        kwargs: Keyword arguments passed to :class:`S4DLayer`.
    """

    def __init__(self, channels: int, **kwargs):
        super().__init__()

        self.l2r = S4DLayer(channels, **kwargs)
        self.r2l = S4DLayer(channels, **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        return torch.cat((
            self.l2r(x),
            self.r2l(x.flip(-1)).flip(-1),
        ), dim=-2)


class S2SNN(nn.Sequential):
    r"""Creates a sequence-to-sequence neural network (S2SNN).

    Arguments:
        in_channels: The number of input channels.
        out_channels: The number of output channels.
        hidden_channels: The number of hidden channels.
        hidden_blocks: The number of hidden blocks. Each block consists in an optional
            normalization, a S4D block, an activation and a 1 by 1 convolution.
        activation: The activation function constructor.
        normalize: Whether channels are normalized or not.
        kwargs: Keyword arguments passed to :class:`S4DBlock`.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int = 64,
        hidden_blocks: int = 3,
        activation: Callable[[], nn.Module] = nn.ReLU,
        normalize: bool = False,
        **kwargs,
    ):
        if normalize:
            normalization = lambda: LayerNorm(dim=-2)
        else:
            normalization = nn.Identity

        linear = lambda x, y: nn.Conv1d(x, y, kernel_size=1)

        layers = [linear(in_channels, hidden_channels)]

        for i in range(hidden_blocks):
            layers.append(
                Residual(
                    normalization(),
                    S4DBlock(hidden_channels, **kwargs),
                    activation(),
                    linear(hidden_channels * 2, hidden_channels),
                )
            )

        layers.append(linear(hidden_channels, out_channels))

        super().__init__(*layers)

        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x: Tensor) -> Tensor:
        y = x.reshape(-1, *x.shape[-2:])
        y = super().forward(y)
        y = y.reshape(*x.shape[:-2], *y.shape[-2:])

        return y
