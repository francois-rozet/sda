#!/usr/bin/env python

from dawgz import job, schedule
from typing import *

from components.mcs import *
from components.utils import *

from utils import *


@job(cpus=1, ram='1GB', time='00:05:00')
def simulate():
    chain = NoisyLorenz63(dt=0.025)

    x = chain.prior((4096,))
    x = chain.trajectory(x, length=2048, last=True)
    x = chain.trajectory(x, length=1024)
    x = chain.preprocess(x)
    x = x.transpose(0, 1)

    i = int(0.8 * len(x))
    j = int(0.9 * len(x))

    splits = {
        'train': x[:i],
        'valid': x[i:j],
        'test': x[j:],
    }

    for name, x in splits.items():
        save_data(x, PATH / f'data/{name}.h5')


if __name__ == '__main__':
    schedule(
        simulate,
        name='Data generation',
        backend='slurm',
        settings={'export': 'ALL'},
        env=['conda activate ssm'],
    )