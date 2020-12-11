# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %load_ext autoreload
# %autoreload 1
# %aimport spatialsfs

import matplotlib.pyplot as plt
import numpy as np

import spatialsfs

T = np.arange(0.0, 100.0, 1.0)
rng = np.random.default_rng(1)

# %%prun
sims = spatialsfs.simulate_branching_diffusions(1000, 0.02, ndims=1, rng=rng)

# %%prun
num_alive_at = np.array([[sim.num_alive_at(t) for sim in sims] for t in T])

# %%prun
positions_at = [[sim.positions_at(t, rng) for sim in sims] for t in T]


ndims = 2
s = 0.02
n_reps = 20000
rng = np.random.default_rng(1)
sims = spatialsfs.simulate_branching_diffusions(n_reps, s, ndims=ndims, rng=rng)

bins = np.arange(0, 50, 1)
plt.hist([sim.num_max for sim in sims], bins=bins, log=True)

print(max([sim.extinction_time for sim in sims]))
bins = np.arange(0, 100, 1)
plt.hist([sim.extinction_time for sim in sims], bins=bins, histtype="step", log=True)

num_alive_at = np.array([[sim.num_alive_at(t) for sim in sims] for t in T])

means = np.mean(num_alive_at, axis=1)
surviving = np.sum(num_alive_at > 0, axis=1)

expectation = np.exp(-s * T)
p_survive_th = -np.expm1(2 * s * np.exp(-s * T) / (np.exp(-s * T) - 1))
p_survive_th[0] = 1

plt.plot(T, means, ".")
plt.plot(T, expectation, "k")
plt.ylim([0, 1.05])

plt.semilogy(T, surviving / n_reps, ".")
plt.semilogy(T, p_survive_th, "k")

plt.plot(T, means / (surviving / n_reps), ".")
plt.semilogy(T, expectation / p_survive_th, "k")


def plot_branching_diffusion_1D(bd, color="gray"):
    for i in range(bd.num_total):
        plt.plot(
            [bd.birth_times[i], bd.death_times[i]],
            [bd.birth_positions[i], bd.death_positions[i]],
            color=color,
        )


def plot_branching_diffusion_2D(bd, color="gray"):
    """Plot birth positions in 2D, colored by birth times."""
    for i in range(bd.num_total):
        col = plt.cm.viridis(bd.birth_times[i] / bd.extinction_time)
        plt.plot(
            bd.birth_positions[i, 0], bd.birth_positions[i, 1], ".", c=col,
        )


for i in range(1000):
    if sims[i].num_total > 100:
        fig = plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plot_branching_diffusion_2D(sims[i])
        plt.xlim([-20, 20])
        plt.ylim([-20, 20])
        plt.subplot(1, 2, 2)
        plt.plot(T, [sims[i].num_alive_at(u) for u in T])
        plt.show()
