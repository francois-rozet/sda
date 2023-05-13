#!/usr/bin/env python

from dawgz import job, after, schedule
from typing import *

from sda.mcs import *
from sda.utils import *

from utils import *


@job(cpus=1, time='00:01:00')
def mkdir():
    (PATH / 'data').mkdir(parents=True, exist_ok=True)


@after(mkdir)
@job(cpus=1, ram='1GB', time='00:05:00')
def simulate():
    chain = make_chain()

    x = chain.prior((1024,))
    x = chain.trajectory(x, length=1024, last=True)
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
        env=['conda activate sda'],
    )
