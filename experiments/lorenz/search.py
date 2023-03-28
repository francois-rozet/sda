#!/usr/bin/env python

import numpy as np
import wandb

from dawgz import job, schedule
from typing import *

from components.mcs import *
from components.score import *
from components.utils import *

from utils import *


CONFIGS = {
    # Architecture
    'embedding': [16, 32, 64],
    'width': [128, 192, 256],
    'depth': [3, 5, 7],
    'activation': ['ELU', 'SiLU'],
    # Training
    'epochs': [256, 384, 512, 768, 1024],
    'epoch_size': [65536],
    'batch_size': [64, 128, 256, 512],
    'optimizer': ['AdamW'],
    'learning_rate': np.geomspace(1e-3, 1e-4).tolist(),
    'weight_decay': np.geomspace(1e-2, 1e-3).tolist(),
    'scheduler': ['linear', 'cosine', 'exponential'],
}


@job(array=64, cpus=2, ram='8GB', time='06:00:00')
def train(i: int):
    config = random_config(CONFIGS)

    run = wandb.init(project='ssm-lorenz-sweep', config=config)
    runpath = PATH / f'sweep/{run.name}_{run.id}'
    runpath.mkdir(parents=True, exist_ok=True)

    save_config(config, runpath)

    # Network
    score = make_score(features=6, **config)
    sde = VPSDE(score, shape=(6,))

    # Data
    trainset = load_data(PATH / 'data/train.h5', window=2)
    validset = load_data(PATH / 'data/valid.h5', window=2)

    # Training
    generator = loop(
        sde,
        trainset,
        validset,
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
    chain = Lorenz63(dt=0.025)

    x = sde.sample((4096,)).unflatten(-1, (-1, 3))
    x = chain.postprocess(x)
    y = chain.transition(x[:, 0])

    mse = (y - x[:, 1]).square().mean().item()

    run.log({'mse': mse})
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
