r"""Lorenz experiment helpers"""

import os

from pathlib import Path
from typing import *

from components.score import *
from components.utils import *


if 'SCRATCH' in os.environ:
    SCRATCH = os.environ['SCRATCH']
    PATH = Path(SCRATCH) / 'ssm/lorenz'
else:
    PATH = Path('.')

PATH.mkdir(parents=True, exist_ok=True)


def make_global_score(
    embedding: int = 32,
    hidden_channels: Sequence[int] = (32, 64),
    hidden_blocks: Sequence[int] = (2, 1),
    activation: str = 'SiLU',
    **absorb,
) -> nn.Module:
    return ScoreUNet(
        channels=3,
        embedding=embedding,
        hidden_channels=hidden_channels,
        hidden_blocks=hidden_blocks,
        activation=ACTIVATIONS[activation],
        spatial=1,
    )


def make_local_score(
    window: int = 5,
    embedding: int = 32,
    width: int = 128,
    depth: int = 5,
    activation: str = 'SiLU',
    **absorb,
) -> nn.Module:
    return ScoreNet(
        features=window * 3,
        embedding=embedding,
        hidden_features=[width] * depth,
        activation=ACTIVATIONS[activation],
    )


def load_score(
    file: Path,
    local: bool = False,
    device: str = 'cpu',
    **kwargs,
) -> nn.Module:
    state = torch.load(file, map_location=device)
    config = load_config(file.parent)
    config.update(kwargs)

    if local:
        score = make_local_score(**config)
    else:
        score = make_global_score(**config)

    score.load_state_dict(state)

    return score
