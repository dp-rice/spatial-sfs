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

from spatialsfs.simulations import load_populations

s = 0.025
pops = load_populations(f'../simulations/branchdiff-s={s:0.3}.pkl.gz')

maxtimes = [pop.get_latest_time() for pop in pops]
mt = max(maxtimes)
bins = np.linspace(0, mt + 1, 100)
plt.hist(maxtimes, bins=bins)
plt.show()
plt.hist(maxtimes, bins=np.arange(100))


def sample(locs, x, sigma):
    return np.sum(np.exp(-(
        (locs - x) / sigma)**2 / 2)) / (np.sqrt(2 * np.pi) * sigma)


# +
nsamples = 100
npops = len(pops)

sigma1 = 4
sigma2 = 0.25
samples1 = np.zeros((nsamples, npops))
samples2 = np.zeros((nsamples, npops))

ts = np.random.exponential(10, size=nsamples)
xs = np.random.normal(0, 10, size=nsamples)
weights = np.zeros((nsamples, npops))

for i in range(nsamples):
    for j, pop in enumerate(pops):
        locs = np.array(pop.locations_at(ts[i]))

        w1 = sample(locs, xs[i], sigma1)
        samples1[i, j] = w1

        w2 = sample(locs, xs[i], sigma2)
        samples2[i, j] = w2

    weights[i, :] = np.exp(ts[i] / 10) * np.exp((xs[i] / 10)**2 / 2)
# -

bins = np.arange(0.0, 10, 0.05)

# +
plt.hist(samples1.flatten(), weights=weights.flatten(), bins=bins, log=True)
plt.show()

plt.hist(samples2.flatten(), weights=weights.flatten(), bins=bins, log=True)
plt.show()

# +
plt.hist(samples1.flatten(),
         weights=weights.flatten(),
         bins=bins,
         cumulative=True,
         density=True,
         histtype='step')

plt.hist(samples2.flatten(),
         weights=weights.flatten(),
         bins=bins,
         cumulative=True,
         density=True,
         histtype='step')
plt.ylim([0.99, 1])

# +
nsamples = 100
npops = len(pops)

bins = np.arange(0.0, 10, 0.05)
sigmas = np.logspace(-2, 2, 5, base=2)

ts = np.random.exponential(10, size=nsamples)
xs = np.random.normal(0, 10, size=nsamples)
weights = np.zeros((nsamples, npops))

for sigma in sigmas:
    print(sigma)
    samples = np.zeros((nsamples, npops))

    for i in range(nsamples):
        for j, pop in enumerate(pops):
            locs = np.array(pop.locations_at(ts[i]))
            w = sample(locs, xs[i], sigma)
            samples[i, j] = w
            weights[i, :] = np.exp(ts[i] / 10) * np.exp((xs[i] / 10)**2 / 2)

    plt.hist(samples.flatten(),
             weights=weights.flatten(),
             bins=bins,
             cumulative=True,
             density=True,
             histtype='step',
             label=sigma)

plt.legend()
plt.ylim([0.99, 1])
# -
