#!/usr/bin/env python

import h5py
import numpy as np
import random

from dawgz import job, after, ensure, schedule
from typing import *

from sda.mcs import KolmogorovFlow

from utils import *


@ensure(lambda i: (PATH / f'data/x_{i:06d}.npy').exists())
@job(array=1024, cpus=1, ram='1GB', time='00:05:00')
def simulate(i: int):
    chain = make_chain()

    random.seed(i)

    x = chain.prior()
    x = chain.trajectory(x, length=128)
    x = x[64:]

    np.save(PATH / f'data/x_{i:06d}.npy', x)


@after(simulate)
@job(cpus=1, ram='1GB', time='00:15:00')
def aggregate():
    files = sorted(PATH.glob('data/x_*.npy'))
    length = len(files)

    i = int(0.8 * length)
    j = int(0.9 * length)

    splits = {
        'train': files[:i],
        'valid': files[i:j],
        'test': files[j:],
    }

    for name, files in splits.items():
        with h5py.File(PATH / f'data/{name}.h5', mode='w') as f:
            f.create_dataset(
                'x',
                shape=(len(files), 64, 2, 64, 64),
                dtype=np.float32,
            )

            for i, x in enumerate(map(np.load, files)):
                f['x'][i] = KolmogorovFlow.coarsen(torch.from_numpy(x), 4)


if __name__ == '__main__':
    (PATH / 'data').mkdir(parents=True, exist_ok=True)

    schedule(
        aggregate,
        name='Data generation',
        backend='slurm',
        prune=True,
        export='ALL',
    )
