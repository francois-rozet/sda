#!/usr/bin/env python

import wandb

from dawgz import job, schedule
from typing import *

from sda.score import *
from sda.utils import *

from utils import *


CONFIG = {
    # Architecture
    'embedding': 64,
    'hidden_channels': (16, 32, 64, 128),
    'hidden_blocks': (5, 3, 2, 1),
    'kernel_size': 3,
    'activation': 'SiLU',
    # Training
    'epochs': 1024,
    'batch_size': 8,
    'optimizer': 'AdamW',
    'learning_rate': 2e-4,
    'weight_decay': 1e-3,
    'scheduler': 'linear',
}


@job(array=3, cpus=4, gpus=1, ram='64GB', time='72:00:00', partition='a5000')
def train(i: int):
    run = wandb.init(project='sda-quasigeostrophic', config=CONFIG)
    runpath = PATH / f'runs/{run.name}_{run.id}'
    runpath.mkdir(parents=True, exist_ok=True)

    save_config(CONFIG, runpath)

    # Network
    score = make_score(**CONFIG)
    sde = VPSDE(score, shape=(13, 6, 128, 128)).cuda()

    # Data
    trainset = RandomCropDataset(PATH / 'data/train.h5', window=13, crop=128, pad=56)
    validset = RandomCropDataset(PATH / 'data/valid.h5', window=13, crop=128, pad=56)

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
    sde = VPSDE(score, shape=(16, 6, 256, 256)).cuda()

    c = trainset.grid.cuda()
    x = sde.sample((2,), c=c, steps=64).cpu()
    q = x[:, ::4, 0]

    run.log({'samples': wandb.Image(draw(q))})
    run.finish()


if __name__ == '__main__':
    schedule(
        train,
        name='Training',
        backend='slurm',
        export='ALL',
        env=['export WANDB_SILENT=true'],
    )
