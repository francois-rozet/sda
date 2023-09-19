#!/usr/bin/env python

import h5py
import pyqg
import xarray

from dawgz import job, after, ensure, schedule
from tqdm import tqdm
from typing import *

from utils import *


@ensure(lambda i: (PATH / f'data/x_{i:06d}.nc').exists())
@job(array=1024, cpus=1, ram='4GB', time='12:00:00')
def simulate(i: int):
    np.random.seed(i)

    hour = 60 * 60
    day = 24 * hour
    year = 360 * day

    model = pyqg.QGModel(
        nx=512,
        dt=hour / 4,
        tmax=5 * year + 60 * day,
    )

    model.diagnostics = {}

    x = []

    for t in model.run_with_snapshots(tsnapstart=5 * year, tsnapint=day):
        ds = model.to_dataset()
        ds = xarray.Dataset({'q': ds.q, 'u': ds.u, 'v': ds.v})
        ds = ds.coarsen(x=2, y=2).mean()

        x.append(ds)

    x = xarray.concat(x, dim='time')

    x.to_netcdf(PATH / f'data/x_{i:06d}.nc')


@after(simulate)
@job(cpus=4, ram='16GB', time='06:00:00')
def aggregate():
    files = sorted(PATH.glob('data/x_*.nc'))
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
                shape=(len(files), 32, 6, 256, 256),
                dtype=np.float32,
            )

            for i, x in enumerate(map(xarray.open_dataset, tqdm(files, ncols=88))):
                x = x.isel(time=slice(0, 32))

                f['x'][i] = np.stack((
                    x.q.isel(lev=0) * 1e5,
                    x.u.isel(lev=0) * 20.0,
                    x.v.isel(lev=0) * 20.0,
                    x.q.isel(lev=1) * 1e6,
                    x.u.isel(lev=1) * 100.0,
                    x.v.isel(lev=1) * 100.0,
                ), axis=1)


if __name__ == '__main__':
    (PATH / 'data').mkdir(parents=True, exist_ok=True)

    schedule(
        aggregate,
        name='Data generation',
        backend='slurm',
        prune=True,
        export='ALL',
    )
