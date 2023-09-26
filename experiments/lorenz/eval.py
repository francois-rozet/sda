#!/usr/bin/env python

import h5py
import numpy as np

from dawgz import job, after, context, ensure, schedule
from typing import *

from sda.mcs import *
from sda.score import *
from sda.utils import *

from utils import *


@ensure(lambda: (PATH / f'results/obs.h5').exists())
@job(cpus=1, ram='1GB', time='00:05:00')
def observations():
    with h5py.File(PATH / 'data/test.h5', mode='r') as f:
        x = f['x'][:, :65]

    y_lo = np.random.normal(x[:, ::8, :1], 0.05)
    y_hi = np.random.normal(x[:, :, :1], 0.25)

    with h5py.File(PATH / 'results/obs.h5', mode='w') as f:
        f.create_dataset('lo', data=y_lo)
        f.create_dataset('hi', data=y_hi)


jobs = []

for name, local in [
    ('polar-capybara-13_y1g6w4jm', True),  # k=1
    ('snowy-leaf-29_711r6as1', True),  # k=2
    ('ruby-serenity-42_nbhxlnf9', True),  # k=3
    ('light-moon-51_09a36gw8', True),  # k=4
    ('lilac-bush-61_7f0sioiw', False),  # k≈8
]:
    for freq in ['lo', 'hi']:
        @after(observations)
        @context(name=name, local=local, freq=freq)
        @job(name=f'{name}_{freq}', array=64, cpus=2, gpus=1, ram='8GB', time='01:00:00')
        def evaluation(i: int):
            chain = make_chain()

            # Observation
            with h5py.File(PATH / 'results/obs.h5', mode='r') as f:
                y = torch.from_numpy(f[freq][i])

            A = lambda x: chain.preprocess(x)[..., :1]

            if freq == 'lo':  # low frequency & low noise
                sigma, step = 0.05, 8
            else:             # high frequency & high noise
                sigma, step = 0.25, 1

            # Ground truth
            x = posterior(y, A=A, sigma=sigma, step=step)[:1024]
            x_ = posterior(y, A=A, sigma=sigma, step=step)[:1024]

            log_px = log_prior(x).mean().item()
            log_py = log_likelihood(y, x, A=A, sigma=sigma, step=step).mean().item()
            w1 = emd(x, x_).item()

            with open(PATH / f'results/stats_{freq}.csv', mode='a') as f:
                f.write(f'{i},ground-truth,,{log_px},{log_py},{w1}\n')

            print('GT:', log_px, log_py, w1, flush=True)

            # Score
            score = load_score(PATH / f'runs/{name}/state.pth', local=local)
            sde = VPSDE(
                GaussianScore(
                    y=y,
                    A=lambda x: x[..., ::step, :1],
                    std=sigma,
                    sde=VPSDE(score, shape=()),
                    gamma=3e-2,
                ),
                shape=(65, 3),
            ).cuda()

            for C in (0, 1, 2, 4, 8, 16):
                x = sde.sample((1024,), steps=256, corrections=C, tau=0.25).cpu()
                x = chain.postprocess(x)

                log_px = log_prior(x).mean().item()
                log_py = log_likelihood(y, x, A=A, sigma=sigma, step=step).mean().item()
                w1 = emd(x, x_).item()

                with open(PATH / f'results/stats_{freq}.csv', mode='a') as f:
                    f.write(f'{i},{name},{C},{log_px},{log_py},{w1}\n')

                print(f'{C:02d}:', log_px, log_py, w1, flush=True)

        jobs.append(evaluation)


if __name__ == '__main__':
    (PATH / 'results').mkdir(parents=True, exist_ok=True)

    schedule(
        *jobs,
        name='Evaluation',
        backend='slurm',
        prune=True,
        export='ALL',
    )
