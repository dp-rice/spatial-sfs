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
pops = load_populations(f"../simulations/branchdiff-s={s:0.3}.pkl.gz")

# Survival times

maxtimes = [pop.get_latest_time() for pop in pops]
mt = max(maxtimes)
bins = np.linspace(0, mt + 1, 100)
plt.hist(maxtimes, bins=bins)
plt.show()
plt.hist(maxtimes, bins=np.arange(100))
print(mt)

# Sampling functions


# +
def gaussian_sum(locs, x, sigmas):
    return np.sum(
        np.exp(-(((locs[:, None] - x) / sigmas[None, :]) ** 2) / 2), axis=0
    ) / (np.sqrt(2 * np.pi) * sigmas)


def sample_populations(populations, nsamples, sigmas):
    npops = len(populations)
    nsigmas = len(sigmas)

    ts = np.random.exponential(10, size=nsamples)
    xs = np.random.normal(0, 10, size=nsamples)
    weights = np.exp(ts / 10) * np.exp((xs / 10) ** 2 / 2)
    weights = np.tile(weights, npops)

    samples = np.zeros((nsigmas, nsamples * npops))
    for i in range(nsamples):
        for j, pop in enumerate(populations):
            locs = np.array(pop.locations_at(ts[i]))
            total = gaussian_sum(locs, xs[i], sigmas)
            samples[:, i + j * nsamples] = total
    return samples, weights


# -

# Theoretical critical time and distance

tc = 1 / (2 * s)
dc = np.sqrt(tc)
print(tc, dc)

# Sample from simulations

np.random.seed(102)
nsamples = 100
sigmas = dc * np.logspace(-5, 5, 11, base=2)
samples, weights = sample_populations(pops, nsamples, sigmas)

# Empirical distribution functions showing discovery/dilution effects

bins = np.arange(0.0, 6, 0.01)
for i, sigma in enumerate(sigmas):
    plt.hist(
        samples[i],
        weights=weights,
        bins=bins,
        cumulative=-1,
        density=True,
        histtype="step",
        label=sigma / dc,
    )
plt.ylim([1e-5, 1])
plt.yscale("log")
plt.legend()

bins = np.arange(0.0, 6, 0.01)
for i, sigma in enumerate(sigmas):
    plt.hist(
        samples[i],
        weights=weights,
        bins=bins,
        cumulative=-1,
        density=True,
        histtype="step",
        label=sigma / dc,
    )
plt.ylim([1e-6, 1e-2])
plt.yscale("log")
plt.legend()

bins = np.arange(0.0, 6, 0.01)
for i, sigma in enumerate(sigmas):
    plt.hist(
        samples[i],
        weights=weights,
        bins=bins,
        cumulative=-1,
        density=True,
        histtype="step",
        label=sigma / dc,
    )
plt.ylim([1e-5, 1])
plt.xlim([1e-2, 6])
plt.yscale("log")
plt.xscale("log")
plt.legend()

# A gamma distribution for comparison

np.random.seed(101)
a = 0.02
rand_gamma = np.random.gamma(a, size=10000)
x = bins
plt.hist(rand_gamma, bins=bins, cumulative=-1, density=True, histtype="step")
plt.semilogy(x, gamma.sf(x, a))
plt.ylim([1e-4, 1])

# ## Compress zeros (depricated)

nzeros = np.sum(samples == 0, axis=1)
print(nzeros / len(samples[0]))


def compress_zeros(samples, weights):
    zeros = samples[-1] == 0
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
    plt.hist(
        new_samples[i],
        weights=new_weights,
        bins=bins,
        cumulative=-1,
        density=True,
        histtype="step",
        label=sigma / dc,
    )
plt.ylim([1e-5, 1])
plt.yscale("log")
plt.legend()

np.isclose(
    np.average(samples, axis=1, weights=weights),
    np.average(new_samples, axis=1, weights=new_weights),
)

# ## Method of moments

t_surv = np.mean(maxtimes)

# Method of moments is in terms of non-zero entries

zeros = samples[-1] == 0
samples_nonzero = samples[:, ~zeros]
weights_nonzero = weights[~zeros]

# Compute sample moments conditional on non-zero


def kth_moment(samples, weights, k):
    return np.average(samples ** k, axis=1, weights=weights)


m1 = kth_moment(samples_nonzero, weights_nonzero, 1)
m2 = kth_moment(samples_nonzero, weights_nonzero, 2)
m3 = kth_moment(samples_nonzero, weights_nonzero, 3)
plt.loglog(sigmas, m1, ".", label="m1")
plt.loglog(sigmas, m2, ".", label="m2")
plt.loglog(sigmas, m3, ".", label="m3")
plt.legend()

# Method-of-moments parameter estmates

alpha_hat = t_surv * m1 ** 2 / m2
beta_hat = m1 / m2

plt.loglog(sigmas / dc, beta_hat, ".", label=r"$\hat \beta$ (selection factor)")
plt.loglog(sigmas / dc, alpha_hat, ".", label=r"$\hat \alpha$ (mutation factor)")
plt.legend()

# ## Goodness-of-fit

# Moment comparisons: in gamma distribution, $m_3 \propto m_2^2 / m_1$

plt.loglog(sigmas, m3, ".")
plt.loglog(sigmas, m2 ** 2 / m1, "x")

# ### KS-Statistics

# Adding in "mutation supply rate" $\theta$


def augment_zeros(samples, weights, t_surv, theta):
    w_0 = (1 / (theta * t_surv) - 1) * np.sum(weights)
    new_samples = np.zeros((samples.shape[0], samples.shape[1] + 1))
    new_samples[:, :-1] = samples
    new_weights = np.zeros(len(weights) + 1)
    new_weights[:-1] = weights
    new_weights[-1] = w_0
    return new_samples, new_weights


theta = 1e-3
samples_aug, weights_aug = augment_zeros(
    samples_nonzero, weights_nonzero, t_surv, theta
)
print(weights_aug[-1], np.sum(weights))

# Check that we've matched moments correctly

print(
    np.isclose(
        np.average(samples_aug, weights=weights_aug, axis=1),
        theta * alpha_hat / beta_hat,
    )
)
print(
    np.isclose(
        np.average(samples_aug ** 2, weights=weights_aug, axis=1),
        theta * alpha_hat / beta_hat ** 2,
    )
)

# Compare survival functions

bins = np.arange(0.0, 6, 0.01)
for i in range(3, 8):
    plt.hist(
        samples_aug[i],
        weights=weights_aug,
        bins=bins,
        cumulative=-1,
        density=True,
        histtype="step",
        log=True,
    )
    plt.semilogy(x, gamma.sf(x, theta * alpha_hat[i], scale=1 / beta_hat[i]))
    plt.show()

i = 5
hist = np.histogram(samples_aug[i], weights=weights_aug, bins=bins)[0]
sf_obs = 1 - np.cumsum(hist) / np.sum(hist)
sf_exp = gamma.sf(bins[1:], theta * alpha_hat[i], scale=1 / beta_hat[i])

# +
ax = plt.subplot(221)
ax.plot(bins[1:], sf_obs)
ax.plot(bins[1:], sf_exp)

ax = plt.subplot(222)
ax.semilogx(bins[1:], sf_obs)
ax.semilogx(bins[1:], sf_exp)
ax.set_yticklabels([])

ax = plt.subplot(223)
ax.plot(bins[1:], np.abs(sf_obs - sf_exp))

ax = plt.subplot(224)
ax.semilogx(bins[1:], np.abs(sf_obs - sf_exp))
ax.set_yticklabels([])

# -


def survival_functions(samples, weights, bins, alpha, beta):
    hist = np.histogram(samples, weights=weights, bins=bins)[0]
    sf_obs = 1 - np.cumsum(hist) / np.sum(hist)
    sf_exp = gamma.sf(bins[1:], alpha, scale=1 / beta)
    return sf_obs, sf_exp


def compare_survival_functions(bins, sf_obs, sf_exp):
    fig = plt.figure()
    ax = fig.add_subplot(221)
    ax.plot(bins[1:], sf_obs)
    ax.plot(bins[1:], sf_exp)

    ax = fig.add_subplot(222)
    ax.semilogx(bins[1:], sf_obs)
    ax.semilogx(bins[1:], sf_exp)
    ax.set_yticklabels([])

    ax = fig.add_subplot(223)
    ax.plot(bins[1:], np.abs(sf_obs - sf_exp))

    ax = fig.add_subplot(224)
    ax.semilogx(bins[1:], np.abs(sf_obs - sf_exp))
    ax.set_yticklabels([])

    return fig


# Compare the EDF to the fitted gamma distribution

bins = np.arange(0.0, 6, 0.001)
for i in range(len(sigmas)):
    sf_obs, sf_exp = survival_functions(
        samples_aug[i], weights_aug, bins, alpha_hat[i] * theta, beta_hat[i]
    )
    fig = compare_survival_functions(bins, sf_obs, sf_exp)
    fig.suptitle(sigmas[i] / dc)
    plt.show()

# Try a statistic that weights the right tail

bins = np.arange(0.0, 6, 0.001)
for i in range(len(sigmas)):
    sf_obs, sf_exp = survival_functions(
        samples_aug[i], weights_aug, bins, alpha_hat[i] * theta, beta_hat[i]
    )
    plt.title(sigmas[i] / dc)
    plt.plot(bins[1:], np.abs(sf_obs - sf_exp) / sf_exp)
    plt.show()
