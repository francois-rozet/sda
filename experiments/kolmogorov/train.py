#!/usr/bin/env python

import wandb

from dawgz import job, schedule
from typing import *

from components.mcs import *
from components.score import *
from components.utils import *

from utils import *


CONFIG = {
    # Architecture
    'embedding': 64,
    'hidden_channels': (64, 128, 256),
    'hidden_blocks': (3, 3, 3),
    'kernel_size': 3,
    'activation': 'SiLU',
    # Training
    'epochs': 1024,
    'batch_size': 32,
    'optimizer': 'AdamW',
    'learning_rate': 2e-4,
    'weight_decay': 1e-3,
    'scheduler': 'linear',
}


@job(array=3, cpus=2, gpus=1, ram='16GB', time='24:00:00')
def train(i: int):
    run = wandb.init(project='ssm-kolmogorov', config=CONFIG)
    runpath = PATH / f'runs/{run.name}_{run.id}'
    runpath.mkdir(parents=True, exist_ok=True)

    save_config(CONFIG, runpath)

    # Network
    score = make_score(window=3, **CONFIG)
    sde = VPSDE(score, shape=(6, 64, 64)).cuda()

    # Data
    trainset = load_data(PATH / 'data/train.h5', window=3)
    validset = load_data(PATH / 'data/valid.h5', window=3)

    # Training
    generator = loop(
        sde,
        trainset,
        validset,
        device='cuda',
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

    # Evaluation
    x = sde.sample((2,), steps=64).cpu()
    x = x.unflatten(1, (-1, 2))
    w = KolmogorovFlow.vorticity(x)

    run.log({'samples': wandb.Image(draw(w))})
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
