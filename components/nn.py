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
