# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %load_ext autoreload
# %autoreload 1
# %aimport spatialsfs.simulations

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gamma

from spatialsfs.simulations import load_populations

# # Fixing the importance sampler
# Previously, our spatial importance sampler for the birth location of a
# mutation was not broad enough. It meant that we never sampled mutations far
# from the center of the sampling kernel. This was presumably undercounting the
# discovery effect.

# ## Load data

s = 0.025
pops = load_populations(f'../simulations/branchdiff-s={s:0.3}.pkl.gz')

# Survival times

maxtimes = [pop.get_latest_time() for pop in pops]
t_surv = np.mean(maxtimes)
print(t_surv)

# Theoretical critical time and distance

tc = 1 / (2 * s)
dc = np.sqrt(tc)
print(tc, dc)

# ## Sampling functions


# +
def gaussian_sum(locs, x, sigmas):
    return np.sum(np.exp(-((locs[:, None] - x) / sigmas[None, :])**2 / 2),
                  axis=0) / (np.sqrt(2 * np.pi) * sigmas)


def trim_zeros(samples, weights):
    zeros = np.all(samples == 0, axis=0)
    samples_nonzero = samples[:, ~zeros]
    weights_nonzero = weights[~zeros]
    return samples_nonzero, weights_nonzero


def sample_populations(populations, nsamples, sigmas, tscale=10, xscale=10):
    npops = len(populations)
    nsigmas = len(sigmas)

    ts = np.random.exponential(tscale, size=nsamples)
    xs = np.random.normal(0, xscale, size=nsamples)
    weights = np.exp(ts / tscale) * np.exp((xs / xscale)**2 / 2)
    weights = np.tile(weights, npops)

    samples = np.zeros((nsigmas, nsamples * npops))
    for i in range(nsamples):
        for j, pop in enumerate(populations):
            locs = np.array(pop.locations_at(ts[i]))
            total = gaussian_sum(locs, xs[i], sigmas)
            samples[:, i + j * nsamples] = total

    return trim_zeros(samples, weights)


def kth_moment(samples, weights, k):
    return np.average(samples**k, axis=1, weights=weights)


def plot_moments(sigmas, samples, weights):
    m1 = kth_moment(samples, weights, 1)
    m2 = kth_moment(samples, weights, 2)
    m3 = kth_moment(samples, weights, 3)
    plt.loglog(sigmas, m1, '.', label='m1')
    plt.loglog(sigmas, m2, '.', label='m2')
    plt.loglog(sigmas, m3, '.', label='m3')
    plt.legend()


# -

# Sample from simulations

# Compute sample moments conditional on non-zero

nsamples = 100
sigmas = dc * np.logspace(-5, 5, 11, base=2)

np.random.seed(102)
for xscale in np.logspace(1, 2, 4):
    np.random.seed(102)
    samples, weights = sample_populations(pops,
                                          nsamples,
                                          sigmas,
                                          xscale=xscale)
    plot_moments(sigmas, samples, weights)
    plt.title(xscale)
    plt.show()
