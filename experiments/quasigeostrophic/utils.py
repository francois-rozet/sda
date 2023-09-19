r"""Quasi-geostrophic helpers"""

import numpy as np
import os
import seaborn

from einops import rearrange
from numpy.typing import ArrayLike
from pathlib import Path
from PIL import Image
from torch import Tensor
from typing import *

from sda.nn import *
from sda.score import *
from sda.utils import *


if 'SCRATCH' in os.environ:
    SCRATCH = os.environ['SCRATCH']
    PATH = Path(SCRATCH) / 'sda/quasigeostrophic'
else:
    PATH = Path('.')

PATH.mkdir(parents=True, exist_ok=True)


class RandomCropDataset(TrajectoryDataset):
    def __init__(self, file: Path, crop: int = None, pad: int = None, **kwargs):
        super().__init__(file, **kwargs)

        *_, H, W = self.data.shape

        grid = torch.stack(
            torch.meshgrid(
                2 * torch.pi * torch.arange(H) / H,
                2 * torch.pi * torch.arange(W) / W,
                indexing='ij',
            )
        )

        self.grid = torch.cat((grid.cos(), grid.sin()))
        self.crop = crop
        self.pad = pad

    def __getitem__(self, i: int) -> Tuple[Tensor, Dict]:
        x, _ = super().__getitem__(i)
        c = self.grid

        if self.crop is None:
            pass
        else:
            i = torch.randint(0, x.shape[-1], size=()).item()
            j = torch.randint(0, x.shape[-2], size=()).item()

            x = torch.roll(x, (i, j), (-1, -2))
            c = torch.roll(c, (i, j), (-1, -2))

            x = x[..., :self.crop, :self.crop]
            c = c[..., :self.crop, :self.crop]

        if self.pad is None:
            return x, dict(c=c)
        else:
            w = torch.zeros_like(x)
            w[..., self.pad:-self.pad, self.pad:-self.pad] = 1.0

            return x, dict(c=c, w=w)


class TemporalBlock(ResidualBlock):
    r"""Creates a temporal-wise residual block for raster sequences."""

    def __init__(self, channels: int):
        super().__init__(
            nn.Conv1d(channels, 4 * channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(4 * channels, channels, kernel_size=1, padding=0),
        )

    def forward(self, x: Tensor) -> Tensor:
        L, C, H, W = x.shape

        x = rearrange(x, 'L C H W -> (H W) C L')
        x = super().forward(x)
        x = rearrange(x, '(H W) C L -> L C H W', H=H, W=W)

        return x


class GlobalScoreUNet(ScoreUNet):
    r"""Creates a U-Net score network for raster sequences."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        for head in self.network.heads[1:]:
            for layer in head:
                channels = getattr(layer, 'out_channels', None)

            head.append(Checkpoint(TemporalBlock(channels)))

        for tail in self.network.tails[:-1]:
            for layer in tail:
                channels = getattr(layer, 'in_channels', None)

            tail.insert(0, Checkpoint(TemporalBlock(channels)))

    def forward(
        self,
        x: Tensor,  # (B, L, C, H, W)
        t: Tensor,  # (B) or ()
        c: Tensor,  # (B, C', H, W) or (C', H, W)
    ) -> Tensor:
        f = super().forward

        if t.shape:
            f = torch.vmap(f)
        else:
            f = torch.vmap(f, in_dims=(0, None, None))

        return f(x, t, c)


def make_score(
    embedding: int = 64,
    hidden_channels: Sequence[int] = (16, 32, 64),
    hidden_blocks: Sequence[int] = (3, 2, 1),
    kernel_size: int = 3,
    activation: str = 'SiLU',
    **absorb,
) -> nn.Module:
    return GlobalScoreUNet(
        channels=6,
        context=4,
        embedding=embedding,
        hidden_channels=hidden_channels,
        hidden_blocks=hidden_blocks,
        kernel_size=kernel_size,
        activation=ACTIVATIONS[activation],
        spatial=2,
        padding_mode='circular',
    )


def load_score(
    file: Path,
    device: str = 'cpu',
    **kwargs,
) -> nn.Module:
    state = torch.load(file, map_location=device)
    config = load_config(file.parent)
    config.update(kwargs)

    score = make_score(**config)
    score.load_state_dict(state)

    return score


def vorticity(x: Tensor) -> Tensor:
    *batch, _, H, W = x.shape

    y = x.reshape(-1, 2, H, W)
    y = torch.nn.functional.pad(y, pad=(1, 1, 1, 1), mode='circular')

    du, = torch.gradient(y[:, 0], dim=-2)
    dv, = torch.gradient(y[:, 1], dim=-1)

    w = dv - du
    w = w[:, 1:-1, 1:-1]
    w = w.reshape(*batch, H, W)

    return w


def vorticity2rgb(
    w: ArrayLike,
    vmin: float = -3.0,
    vmax: float = 3.0,
) -> ArrayLike:
    w = np.asarray(w)
    w = (w - vmin) / (vmax - vmin)
    w = 2 * w - 1
    w = np.sign(w) * np.abs(w) ** 0.8
    w = (w + 1) / 2
    w = seaborn.cm.icefire(w)
    w = 256 * w[..., :3]
    w = w.astype(np.uint8)

    return w


def draw(
    w: ArrayLike,
    mask: ArrayLike = None,
    pad: int = 16,
    zoom: int = 1,
    **kwargs,
) -> Image.Image:
    w = vorticity2rgb(w, **kwargs)
    w = w[(None,) * (5 - w.ndim)]

    M, N, H, W, _ = w.shape

    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        mask = mask[(None,) * (4 - mask.ndim)]

    img = Image.new(
        'RGB',
        size=(
            N * (W + pad) + pad,
            M * (H + pad) + pad,
        ),
        color=(255, 255, 255),
    )

    for i in range(M):
        for j in range(N):
            offset = (
                j * (W + pad) + pad,
                i * (H + pad) + pad,
            )

            img.paste(Image.fromarray(w[i][j]), offset)

            if mask is not None:
                img.paste(
                    Image.new('L', size=(W, H), color=240),
                    offset,
                    Image.fromarray(~mask[i][j]),
                )

    if zoom > 1:
        return img.resize((img.width * zoom, img.height * zoom), resample=0)
    else:
        return img


def save_gif(
    w: ArrayLike,
    file: Path,
    dt: float = 0.2,
    **kwargs,
) -> None:
    w = vorticity2rgb(w, **kwargs)

    imgs = [Image.fromarray(img) for img in w]
    imgs[0].save(
        file,
        save_all=True,
        append_images=imgs[1:],
        duration=int(1000 * dt),
        loop=0,
    )
