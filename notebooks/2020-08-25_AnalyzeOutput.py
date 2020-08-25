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

s = 0.025
pops = load_populations(f'../simulations/branchdiff-s={s:0.3}.pkl.gz')

maxtimes = [pop.get_latest_time() for pop in pops]
mt = max(maxtimes)
bins = np.linspace(0, mt + 1, 100)
plt.hist(maxtimes, bins=bins)
plt.show()
plt.hist(maxtimes, bins=np.arange(100))
print(mt)


# +
def gaussian_sum(locs, x, sigmas):
    return np.sum(np.exp(-((locs[:, None] - x) / sigmas[None, :])**2 / 2),
                  axis=0) / (np.sqrt(2 * np.pi) * sigmas)


def sample_populations(populations, nsamples, sigmas):
    npops = len(populations)
    nsigmas = len(sigmas)

    ts = np.random.exponential(10, size=nsamples)
    xs = np.random.normal(0, 10, size=nsamples)
    weights = np.exp(ts / 10) * np.exp((xs / 10)**2 / 2)
    weights = np.tile(weights, npops)

    samples = np.zeros((nsigmas, nsamples * npops))
    for i in range(nsamples):
        for j, pop in enumerate(populations):
            locs = np.array(pop.locations_at(ts[i]))
            total = gaussian_sum(locs, xs[i], sigmas)
            samples[:, i + j * nsamples] = total
    return samples, weights


# -

tc = 1 / (2 * s)
dc = np.sqrt(tc)
print(tc, dc)

np.random.seed(102)
nsamples = 100
sigmas = dc * np.logspace(-2, 2, 5, base=2)
samples, weights = sample_populations(pops, nsamples, sigmas)

bins = np.arange(0.0, 6, 0.01)
for i, sigma in enumerate(sigmas):
    plt.hist(samples[i],
             weights=weights,
             bins=bins,
             cumulative=-1,
             density=True,
             histtype='step',
             label=sigma / dc)
plt.ylim([1e-5, 1])
plt.yscale('log')
plt.legend()

a = 0.02
rand_gamma = np.random.gamma(a, size=10000)
plt.hist(rand_gamma, bins=bins, cumulative=-1, density=True, histtype='step')
plt.yscale('log')

x = bins
plt.hist(rand_gamma, bins=bins, cumulative=-1, density=True, histtype='step')
plt.semilogy(x, gamma.sf(x, a))
plt.ylim([1e-4, 1])

# ## Compress zeros

nzeros = np.sum(samples == 0, axis=1)
print(nzeros / len(samples[0]))


def compress_zeros(samples, weights):
    zeros = samples[0] == 0
    nzeros = np.sum(zeros)
    new_length = samples.shape[1] - nzeros + 1
    new_samples = np.zeros((samples.shape[0], new_length))
    new_weights = np.zeros(new_length)
    new_samples[:, 1:] = samples[:, ~zeros]
    new_weights[0] = np.sum(weights[zeros])
    new_weights[1:] = weights[~zeros]
    return new_samples, new_weights


new_samples, new_weights = compress_zeros(samples, weights)

bins = np.arange(0.0, 6, 0.01)
for i, sigma in enumerate(sigmas):
    plt.hist(new_samples[i],
             weights=new_weights,
             bins=bins,
             cumulative=-1,
             density=True,
             histtype='step',
             label=sigma / dc)
plt.ylim([1e-5, 1])
plt.yscale('log')
plt.legend()

np.average(samples, axis=1, weights=weights)

np.average(new_samples, axis=1, weights=new_weights)
