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


@job(array=64, cpus=2, gpus=1, ram='16GB', time='06:00:00')
def train(i: int):
    config = random_config(CONFIGS)

    run = wandb.init(project='ssm-kolmogorov-sweep', config=config)
    runpath = PATH / f'sweep/{run.name}_{run.id}'
    runpath.mkdir(parents=True, exist_ok=True)

    save_config(config, runpath)

    # Network
    score = make_score(channels=4, **config)
    sde = VPSDE(score, shape=(4, 64, 64)).cuda()

    # Data
    trainset = load_data(PATH / 'data/train.h5', window=2)
    validset = load_data(PATH / 'data/valid.h5', window=2)

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
    x = sde.sample((2,), steps=64).cpu()
    w = w.unflatten(1, (-1, 2))
    w = KolmogorovFlow.vorticity(x)

    run.log({'samples': wandb.Image(draw(w))})
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
