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


def make_score(
    window: int,
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


def load_score(file: Path, device: str = 'cpu', **kwargs) -> nn.Module:
    state = torch.load(file, map_location=device)
    config = load_config(file.parent)
    config.update(kwargs)

    score = make_score(**config)
    score.load_state_dict(state)

    return score
