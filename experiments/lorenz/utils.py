r"""Lorenz experiment helpers"""

import os

from pathlib import Path
from typing import *

from sda.mcs import *
from sda.score import *
from sda.utils import *


if 'SCRATCH' in os.environ:
    SCRATCH = os.environ['SCRATCH']
    PATH = Path(SCRATCH) / 'sda/lorenz'
else:
    PATH = Path('.')

PATH.mkdir(parents=True, exist_ok=True)


def make_chain() -> MarkovChain:
    return NoisyLorenz63(dt=0.025)


def make_global_score(
    embedding: int = 32,
    hidden_channels: Sequence[int] = (64,),
    hidden_blocks: Sequence[int] = (3,),
    activation: str = 'SiLU',
    **absorb,
) -> nn.Module:
    return MCScoreWrapper(
        ScoreUNet(
            channels=3,
            embedding=embedding,
            hidden_channels=hidden_channels,
            hidden_blocks=hidden_blocks,
            activation=ACTIVATIONS[activation],
            spatial=1,
        )
    )


def make_local_score(
    window: int = 5,
    embedding: int = 32,
    width: int = 128,
    depth: int = 5,
    activation: str = 'SiLU',
    **absorb,
) -> nn.Module:
    return MCScoreNet(
        features=3,
        order=window // 2,
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


def log_prior(x: Tensor) -> Tensor:
    chain = make_chain()

    log_p = chain.log_prob(x[..., :-1, :], x[..., 1:, :])
    log_p = log_p.sum(dim=-1)

    return log_p


def log_likelihood(
    y: Tensor,
    x: Tensor,
    A: Callable[[Tensor], Tensor] = lambda x: x,
    sigma: float = 1.0,
    step: int = 1,
) -> Tensor:
    x = x[..., ::step, :]

    log_p = Normal(y, sigma).log_prob(A(x))
    log_p = log_p.sum(dim=(-1, -2))

    return log_p


def posterior(
    y: Tensor,
    A: Callable[[Tensor], Tensor] = lambda x: x,
    sigma: float = 1.0,
    step: int = 1,
    particles: int = 16384,
) -> Tensor:
    chain = make_chain()

    x = chain.prior((particles,))
    x = chain.trajectory(x, length=64, last=True)

    def likelihood(y, x):
        w = Normal(y, sigma).log_prob(A(x)).sum(dim=-1)
        w = torch.softmax(w, 0)
        return w

    return bpf(x, y, chain.transition, likelihood, step)[:, step:]


def weak_4d_var(
    x: Tensor,
    y: Tensor,
    A: Callable[[Tensor], Tensor] = lambda x: x,
    sigma: float = 1.0,
    step: int = 1,
    iterations: int = 16,
) -> Tensor:
    x_b = x[0]
    x = torch.nn.Parameter(x.clone())
    optimizer = torch.optim.LBFGS((x,))

    def closure():
        optimizer.zero_grad()
        loss = (x[0] - x_b).square().sum() - log_prior(x) - log_likelihood(y, x, A, sigma, step)
        loss.backward()
        return loss

    for _ in range(iterations):
        optimizer.step(closure)

    return x.data
