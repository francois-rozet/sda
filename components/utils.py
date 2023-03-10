r"""Helpers"""

import h5py
import math
import torch

from pathlib import Path
from torch import Tensor
from tqdm import trange
from typing import *

from .score import *


def read(file: Path, window: int = None) -> Tensor:
    with h5py.File(file, mode='r') as f:
        data = f['x'][:]

    data = torch.from_numpy(data)

    if window is None:
        pass
    elif window == 1:
        data = data.flatten(0, 1)
    else:
        data = data.unfold(1, window, 1)
        data = data.movedim(-1, 3)
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
    for epoch in trange(epochs, ncols=88):
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
        yield (
            torch.stack(losses_train).mean().item(),
            torch.stack(losses_valid).mean().item(),
            optimizer.param_groups[0]['lr'],
        )

        ## Step
        scheduler.step()
