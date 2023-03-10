#!/usr/bin/env python

import json
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import seaborn
import wandb

from dawgz import job, schedule
from pathlib import Path
from typing import *

from components.mcs import *
from components.score import *
from components.utils import *


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


def build(channels: int, **config) -> nn.Module:
    activation = {
        'ReLU': nn.ReLU,
        'ELU': nn.ELU,
        'GELU': nn.GELU,
        'SiLU': nn.SiLU,
    }.get(config['activation'])

    return ScoreUNet(
        channels=channels,
        embedding=config['embedding'],
        hidden_channels=config['hidden_channels'],
        hidden_blocks=config['hidden_blocks'],
        kernel_size=config['kernel_size'],
        activation=activation,
        spatial=2,
        padding_mode='circular',
    )


@job(array=64, cpus=2, gpus=1, ram='16GB', time='06:00:00')
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

    # Network
    score = build(channels=2, **config)
    sde = VPSDE(score, shape=(2, 64, 64)).cuda()

    # Data
    trainset = read(PATH / 'data/train.h5', window=1)
    validset = read(PATH / 'data/valid.h5', window=1)

    # Training
    generator = loop(
        sde,
        trainset,
        validset,
        device='cuda',
        **config,
    )

    for loss_train, loss_valid, lr in generator:
        run.log({
            'loss_train': loss_train,
            'loss_valid': loss_valid,
            'lr': lr,
        })

    # Save
    torch.save(
        score.state_dict(),
        runpath / f'state.pth',
    )

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
