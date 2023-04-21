#!/usr/bin/env python

import json
import wandb

from dawgz import job, schedule
from typing import *

from components.mcs import *
from components.score import *
from components.utils import *

from utils import *


GLOBAL_CONFIG = {
    # Architecture
    'embedding': 32,
    'hidden_channels': (32, 64),
    'hidden_blocks': (2, 1),
    'activation': 'SiLU',
    # Training
    'epochs': 1024,
    'epoch_size': 16384,
    'batch_size': 64,
    'optimizer': 'AdamW',
    'learning_rate': 1e-3,
    'weight_decay': 1e-3,
    'scheduler': 'linear',
}

LOCAL_CONFIG = {
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


@job(array=3, cpus=2, ram='8GB', time='06:00:00')
def train_global(i: int):
    run = wandb.init(project='ssm-lorenz-global', config=GLOBAL_CONFIG)
    runpath = PATH / f'runs/{run.name}_{run.id}'
    runpath.mkdir(parents=True, exist_ok=True)

    save_config(GLOBAL_CONFIG, runpath)

    # Network
    score = make_global_score(**GLOBAL_CONFIG)
    sde = VPSDE(score, shape=(3, 16))

    # Data
    trainset = load_data(PATH / 'data/train.h5').unfold(1, 16, 1).flatten(0, 1)
    validset = load_data(PATH / 'data/valid.h5').unfold(1, 16, 1).flatten(0, 1)

    # Training
    generator = loop(
        sde,
        trainset,
        validset,
        **GLOBAL_CONFIG,
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

    x = sde.sample((1024,), steps=64)
    x = x.transpose(-1, -2)
    x = chain.postprocess(x)
    y = chain.transition(x[:, :-1])

    mse = (y - x[:, 1:]).square().mean().item()

    run.log({'mse': mse})
    run.finish()


@job(array=3, cpus=2, ram='8GB', time='06:00:00')
def train_local(i: int):
    run = wandb.init(project='ssm-lorenz-local', config=LOCAL_CONFIG)
    runpath = PATH / f'runs/{run.name}_{run.id}'
    runpath.mkdir(parents=True, exist_ok=True)

    save_config(LOCAL_CONFIG, runpath)

    # Network
    score = make_local_score(window=5, **LOCAL_CONFIG)
    sde = VPSDE(score, shape=(15,))

    # Data
    trainset = load_data(PATH / 'data/train.h5', window=5)
    validset = load_data(PATH / 'data/valid.h5', window=5)

    # Training
    generator = loop(
        sde,
        trainset,
        validset,
        **LOCAL_CONFIG,
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

    x = sde.sample((4096,), steps=64)
    x = x.unflatten(-1, (-1, 3))
    x = chain.postprocess(x)
    y = chain.transition(x[:, :-1])

    mse = (y - x[:, 1:]).square().mean().item()

    run.log({'mse': mse})
    run.finish()


if __name__ == '__main__':
    schedule(
        train_global,
        train_local,
        name='Training',
        backend='slurm',
        settings={'export': 'ALL'},
        env=[
            'conda activate ssm',
            'export WANDB_SILENT=true',
        ],
    )
