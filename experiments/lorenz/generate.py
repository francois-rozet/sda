#!/usr/bin/env python

import h5py

from dawgz import job, after, schedule
from typing import *

from utils import *


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
        with h5py.File(PATH / f'data/{name}.h5', mode='w') as f:
            f.create_dataset('x', data=x, dtype=np.float32)


if __name__ == '__main__':
    (PATH / 'data').mkdir(parents=True, exist_ok=True)

    schedule(
        simulate,
        name='Data generation',
        backend='slurm',
        export='ALL',
    )
