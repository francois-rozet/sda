#!/usr/bin/env python

import json
import wandb

from dawgz import job, schedule
from typing import *

from components.mcs import *
from components.score import *
from components.utils import *

from utils import *


CONFIG = {
    # Architecture
    'embedding': 32,
    'width': 128,
    'depth': 5,
    'activation': 'SiLU',
    # Training
    'epochs': 1024,
    'epoch_size': 65536,
    'batch_size': 256,
    'optimizer': 'AdamW',
    'learning_rate': 1e-3,
    'weight_decay': 1e-3,
    'scheduler': 'linear',
}


@job(array=4, cpus=2, ram='8GB', time='06:00:00')
def train(i: int):
    if i % 2 == 0:
        joint, group = False, 'marginal'
    else:
        joint, group = True, 'joint'

    run = wandb.init(project='ssm-lorenz', group=group, config=CONFIG)
    runpath = PATH / f'runs/{run.name}_{run.id}'
    runpath.mkdir(parents=True, exist_ok=True)

    save_config(CONFIG, runpath)

    # Network
    score = make_score(features=6 if joint else 3, **CONFIG)
    sde = VPSDE(score, shape=(6 if joint else 3,))

    # Data
    trainset = load_data(PATH / 'data/train.h5', window=2 if joint else 1)
    validset = load_data(PATH / 'data/valid.h5', window=2 if joint else 1)

    # Training
    generator = loop(
        sde,
        trainset,
        validset,
        **CONFIG,
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

    run.finish()


if __name__ == '__main__':
    schedule(
        train,
        name='Training',
        backend='slurm',
        settings={'export': 'ALL'},
        env=[
            'conda activate ssm',
            'export WANDB_SILENT=true',
        ],
    )
