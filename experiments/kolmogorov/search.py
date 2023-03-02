#!/usr/bin/env python

import h5py
import json
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import seaborn
import time
import wandb

from dawgz import job, schedule
from pathlib import Path
from tqdm import trange
from typing import *

from components.mcs import *
from components.score import *


SCRATCH = os.environ.get('SCRATCH', '.')
PATH = Path(SCRATCH) / 'ssm/kolmogorov'
PATH.mkdir(parents=True, exist_ok=True)

CONFIG = {
    # Architecture
    'embedding': [64, 128, 256],
    'hidden_channels': [(64, 96, 128, 192)],
    'hidden_blocks': [(3,) * 3, (3,) * 4],
    'kernel_size': [3, 5],
    'activation': ['GELU', 'SiLU'],
    # Training
    'epochs': [256, 384, 512],
    'batch_size': [32, 64],
    'optimizer': ['AdamW'],
    'learning_rate': np.geomspace(1e-3, 1e-4).tolist(),
    'weight_decay': np.geomspace(1e-2, 1e-3).tolist(),
    'scheduler': ['linear', 'cosine', 'exponential'],
}


def build(**config) -> nn.Module:
    activation = {
        'ReLU': nn.ReLU,
        'ELU': nn.ELU,
        'GELU': nn.GELU,
        'SiLU': nn.SiLU,
    }.get(config['activation'])

    return ScoreUNet(
        channels=2,
        embedding=config['embedding'],
        hidden_channels=config['hidden_channels'],
        hidden_blocks=config['hidden_blocks'],
        kernel_size=config['kernel_size'],
        activation=activation,
        spatial=2,
        padding_mode='circular',
    )


def read(file: Path) -> Tensor:
    with h5py.File(file, mode='r') as f:
        data = f['x'][:]

    data = torch.from_numpy(data)
    data = data.reshape(-1, 2, 64, 64)

    return data


@job(array=32, cpus=2, gpus=1, ram='32GB', time='06:00:00')
def train(i: int):
    config = {
        key: random.choice(values)
        for key, values in CONFIG.items()
    }

    run = wandb.init(project='ssm-kolmogorov-sweep', config=config)
    runpath = PATH / f'sweep/{run.name}_{run.id}'
    runpath.mkdir(parents=True, exist_ok=True)

    with open(runpath / 'config.json', 'w') as f:
        json.dump(config, f)

    # Data
    trainset = read(PATH / 'data/train.h5')
    validset = read(PATH / 'data/valid.h5')

    # Network
    score = build(**config)
    sde = VPSDE(score, shape=(2, 64, 64)).cuda()

    # Training
    epochs = config['epochs']
    batch_size = config['batch_size']
    best = 0.05

    ## Optimizer
    if config['optimizer'] == 'AdamW':
        optimizer = torch.optim.AdamW(
            score.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay'],
        )
    else:
        raise ValueError()

    if config['scheduler'] == 'linear':
        lr = lambda t: 1 - (t / epochs)
    elif config['scheduler'] == 'cosine':
        lr = lambda t: (1 + math.cos(math.pi * t / epochs)) / 2
    elif config['scheduler'] == 'exponential':
        lr = lambda t: math.exp(-7 * (t / epochs) ** 2)
    else:
        raise ValueError()

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr)

    ## Loop
    for epoch in trange(epochs, ncols=88):
        losses_train = []
        losses_val = []

        ### Train
        i = np.random.choice(
            len(trainset),
            size=4096,
            replace=False,
        )

        start = time.time()

        for x in trainset[i].cuda().split(batch_size):
            optimizer.zero_grad()
            l = sde.loss(x)
            l.backward()
            optimizer.step()

            losses_train.append(l.detach())

        end = time.time()

        ### Valid
        i = np.random.choice(
            len(validset),
            size=1024,
            replace=False,
        )

        with torch.no_grad():
            for x in validset[i].cuda().split(batch_size):
                losses_val.append(sde.loss(x))

        ### Logs
        loss_train = torch.stack(losses_train).mean().item()
        loss_val = torch.stack(losses_val).mean().item()

        run.log({
            'loss': loss_train,
            'loss_val': loss_val,
            'speed': 1 / (end - start),
            'lr': optimizer.param_groups[0]['lr'],
        })

        ### Checkpoint
        if loss_val < best * 0.95:
            best = loss_val
            torch.save(
                score.state_dict(),
                runpath / f'checkpoint_{epoch:04d}.pth',
            )

        scheduler.step()

    # Load best checkpoint
    checkpoints = sorted(runpath.glob('checkpoint_*.pth'))
    state = torch.load(checkpoints[-1])

    score.load_state_dict(state)

    # Evaluation
    x = sde.sample((4,), steps=64).cpu()
    w = KolmogorovFlow.vorticity(x)

    fig, axs = plt.subplots(2, 2, figsize=(6.4, 6.4))

    for i, ax in enumerate(axs.flat):
        ax.imshow(w[i], cmap=seaborn.cm.icefire)
        ax.label_outer()

    fig.tight_layout()

    run.log({'samples': wandb.Image(fig)})
    run.finish()


if __name__ == '__main__':
    schedule(
        train,
        name='Hyperparameter search',
        backend='slurm',
        settings={'export': 'ALL'},
        env=[
            'conda activate ssm',
            'export WANDB_SILENT=true',
        ],
    )
