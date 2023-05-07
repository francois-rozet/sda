r"""Kolmogorov experiment helpers"""

import os
import seaborn

from numpy.typing import ArrayLike
from pathlib import Path
from PIL import Image, ImageDraw, ImageOps
from typing import *

from components.score import *
from components.utils import *


if 'SCRATCH' in os.environ:
    SCRATCH = os.environ['SCRATCH']
    PATH = Path(SCRATCH) / 'ssm/kolmogorov'
else:
    PATH = Path('.')

PATH.mkdir(parents=True, exist_ok=True)


class SpecialScoreUNet(ScoreUNet):
    r"""Creates a score U-Net with a forcing channel."""

    def __init__(
        self,
        channels: int,
        size: int = 64,
        **kwargs,
    ):
        super().__init__(channels + 1, **kwargs)

        domain = 2 * torch.pi / size * (torch.arange(size) + 1 / 2)
        forcing = torch.sin(4 * domain).expand(1, size, size).clone()

        self.register_buffer('forcing', forcing)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        x, f = broadcast(x, self.forcing, ignore=3)
        x = torch.cat((x, f), dim=-3)

        return super().forward(x, t)[..., :-1, :, :]


def make_score(
    window: int = 3,
    embedding: int = 64,
    hidden_channels: Sequence[int] = (64, 128, 256),
    hidden_blocks: Sequence[int] = (3, 3, 3),
    kernel_size: int = 3,
    activation: str = 'SiLU',
    **absorb,
) -> nn.Module:
    score = MCScoreNet(2, order=window // 2)
    score.kernel = SpecialScoreUNet(
        channels=window * 2,
        embedding=embedding,
        hidden_channels=hidden_channels,
        hidden_blocks=hidden_blocks,
        kernel_size=kernel_size,
        activation=ACTIVATIONS[activation],
        spatial=2,
        padding_mode='circular',
    )

    return score


def load_score(file: Path, device: str = 'cpu', **kwargs) -> nn.Module:
    state = torch.load(file, map_location=device)
    config = load_config(file.parent)
    config.update(kwargs)

    score = make_score(**config)
    score.load_state_dict(state)

    return score


def vorticity2rgb(
    w: ArrayLike,
    vmin: float = -1.25,
    vmax: float = 1.25,
) -> ArrayLike:
    w = np.asarray(w)
    w = (w - vmin) / (vmax - vmin)
    w = seaborn.cm.icefire(w)
    w = 256 * w[..., :3]
    w = w.astype(np.uint8)

    return w


def draw(
    w: ArrayLike,
    mask: ArrayLike = None,
    pad: int = 4,
    zoom: int = 1,
    **kwargs,
) -> Image.Image:
    w = vorticity2rgb(w, **kwargs)
    m, n, width, height, _ = w.shape

    img = Image.new(
        'RGB',
        size=(
            n * (width + pad) + pad,
            m * (height + pad) + pad
        ),
        color=(255, 255, 255),
    )

    for i in range(m):
        for j in range(n):
            offset = (
                j * (width + pad) + pad,
                i * (height + pad) + pad,
            )

            img.paste(Image.fromarray(w[i][j]), offset)

            if mask is not None:
                img.paste(
                    Image.new('L', size=(width, height), color=240),
                    offset,
                    Image.fromarray(~mask[i][j]),
                )

    if zoom > 1:
        return img.resize((img.width * zoom, img.height * zoom), resample=0)
    else:
        return img


def sandwich(
    w: ArrayLike,
    offset: int = 5,
    border: int = 1,
    mirror: bool = False,
    **kwargs,
):
    w = vorticity2rgb(w, **kwargs)
    n, width, height, _ = w.shape

    if mirror:
        w = w[:, :, ::-1]

    img = Image.new(
        'RGB',
        size=(
            width + (n - 1) * offset,
            height + (n - 1) * offset,
        ),
        color=(255, 255, 255),
    )

    draw = ImageDraw.Draw(img)

    for i in range(n):
        draw.rectangle(
            (i * offset - border, i * offset - border, img.width, img.height),
            (255, 255, 255),
        )
        img.paste(Image.fromarray(w[i]), (i * offset, i * offset))

    if mirror:
        return ImageOps.mirror(img)
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
