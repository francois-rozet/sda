r"""Helpers"""

import h5py
import json
import math
import numpy as np
import ot
import random
import torch

from pathlib import Path
from torch import Tensor
from tqdm import trange
from typing import *

from .score import *


ACTIVATIONS = {
    'ReLU': torch.nn.ReLU,
    'ELU': torch.nn.ELU,
    'GELU': torch.nn.GELU,
    'SELU': torch.nn.SELU,
    'SiLU': torch.nn.SiLU,
}


def random_config(configs: Dict[str, Sequence[Any]]) -> Dict[str, Any]:
    return {
        key: random.choice(values)
        for key, values in configs.items()
    }


def save_config(config: Dict[str, Any], path: Path) -> None:
    with open(path / 'config.json', mode='x') as f:
        json.dump(config, f)


def load_config(path: Path) -> Dict[str, Any]:
    with open(path / 'config.json', mode='r') as f:
        return json.load(f)


def save_data(x: Tensor, file: Path) -> None:
    with h5py.File(file, mode='w') as f:
        f.create_dataset('x', data=x, dtype=np.float32)


def load_data(file: Path, window: int = None) -> Tensor:
    with h5py.File(file, mode='r') as f:
        data = f['x'][:]

    data = torch.from_numpy(data)

    if window is None:
        pass
    elif window == 1:
        data = data.flatten(0, 1)
    else:
        data = data.unfold(1, window, 1)
        data = data.movedim(-1, 2)
        data = data.flatten(2, 3)
        data = data.flatten(0, 1)

    return data


def loop(
    sde: VPSDE,
    trainset: Tensor,
    validset: Tensor,
    epochs: int = 256,
    epoch_size: int = 4096,
    batch_size: int = 64,
    optimizer: str = 'AdamW',
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-3,
    scheduler: float = 'linear',
    device: str = 'cpu',
    **absorb,
) -> Iterator:
    # Optimizer
    if optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(
            sde.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    else:
        raise ValueError()

    # Scheduler
    if scheduler == 'linear':
        lr = lambda t: 1 - (t / epochs)
    elif scheduler == 'cosine':
        lr = lambda t: (1 + math.cos(math.pi * t / epochs)) / 2
    elif scheduler == 'exponential':
        lr = lambda t: math.exp(-7 * (t / epochs) ** 2)
    else:
        raise ValueError()

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr)

    # Loop
    for epoch in (bar := trange(epochs, ncols=88)):
        losses_train = []
        losses_valid = []

        ## Train
        sde.train()

        i = torch.randint(len(trainset), (epoch_size,))
        subset = trainset[i].to(device)

        for x in subset.split(batch_size):
            optimizer.zero_grad()
            l = sde.loss(x)
            l.backward()
            optimizer.step()

            losses_train.append(l.detach())

        ## Valid
        sde.eval()

        i = torch.randint(len(validset), (epoch_size,))
        subset = validset[i].to(device)

        with torch.no_grad():
            for x in subset.split(batch_size):
                losses_valid.append(sde.loss(x))

        ## Stats
        loss_train = torch.stack(losses_train).mean().item()
        loss_valid = torch.stack(losses_valid).mean().item()
        lr = optimizer.param_groups[0]['lr']

        yield loss_train, loss_valid, lr

        bar.set_postfix(lt=loss_train, lv=loss_valid, lr=lr)

        ## Step
        scheduler.step()


def bpf(
    x: Tensor,  # (M, *)
    y: Tensor,  # (N, *)
    transition: Callable[[Tensor], Tensor],
    likelihood: Callable[[Tensor, Tensor], Tensor],
    step: int = 1,
) -> Tensor:  # (M, N + 1, *)
    r"""Performs bootstrap particle filter (BPF) sampling

    .. math:: p(x_0, x_1, ..., x_n | y_1, ..., y_n)
        = p(x_0) \prod_i p(x_i | x_{i-1}) p(y_i | x_i)

    Wikipedia:
        https://wikipedia.org/wiki/Particle_filter

    Arguments:
        x: A set of initial states :math:`x_0`.
        y: The vector of observations :math:`(y_1, ..., y_n)`.
        transition: The transition function :math:`p(x_i | x_{i-1})`.
        likelihood: The likelihood function :math:`p(y_i | x_i)`.
        step: The number of transitions per observation.
    """

    x = x[:, None]

    for yi in y:
        for _ in range(step):
            xi = transition(x[:, -1])
            x = torch.cat((x, xi[:, None]), dim=1)

        w = likelihood(yi, xi)
        j = torch.multinomial(w, len(w), replacement=True)
        x = x[j]

    return x


def emd(
    x: Tensor,  # (M, *)
    y: Tensor,  # (N, *)
) -> Tensor:
    r"""Computes the earth mover's distance (EMD) between two distributions.

    Wikipedia:
        https://wikipedia.org/wiki/Earth_mover%27s_distance

    Arguments:
        x: A set of samples :math:`x ~ p(x)`.
        y: A set of samples :math:`y ~ q(y)`.
    """

    return ot.emd2(
        x.new_tensor(()),
        y.new_tensor(()),
        torch.cdist(x.flatten(1), y.flatten(1)),
    )


def mmd(
    x: Tensor,  # (M, *)
    y: Tensor,  # (N, *)
) -> Tensor:
    r"""Computes the empirical maximum mean discrepancy (MMD) between two distributions.

    Wikipedia:
        https://wikipedia.org/wiki/Kernel_embedding_of_distributions

    Arguments:
        x: A set of samples :math:`x ~ p(x)`.
        y: A set of samples :math:`y ~ q(y)`.
    """

    x = x.flatten(1)
    y = y.flatten(1)

    xx = x @ x.T
    yy = y @ y.T
    xy = x @ y.T

    dxx = xx.diag().unsqueeze(1)
    dyy = yy.diag().unsqueeze(0)

    err_xx = dxx + dxx.T - 2 * xx
    err_yy = dyy + dyy.T - 2 * yy
    err_xy = dxx + dyy - 2 * xy

    mmd = 0

    for sigma in (1e-3, 1e-2, 1e-1, 1e-0, 1e1, 1e2, 1e3):
        kxx = torch.exp(-err_xx / sigma)
        kyy = torch.exp(-err_yy / sigma)
        kxy = torch.exp(-err_xy / sigma)

        mmd = mmd + kxx.mean() + kyy.mean() - 2 * kxy.mean()

    return mmd
