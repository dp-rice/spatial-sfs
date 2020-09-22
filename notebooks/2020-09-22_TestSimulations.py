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
from matplotlib.colors import LogNorm

import spatialsfs


def plot_branching_diffusion_2D(bd):
    """Plot birth positions in 2D, colored by birth times."""
    for i in range(len(bd)):
        col = plt.cm.viridis(
            bd.branching_process.birth_times[i] / bd.branching_process.final_time
        )
        plt.plot(
            bd.birth_positions[i, 0], bd.birth_positions[i, 1], ".", c=col,
        )


# %%prun
seed = 100
seed1, seed2 = np.random.SeedSequence(seed).spawn(2)
num_steps = 10000
num_samples = 10000
s = 0.02
ndim = 2
d = 1.0
concentration = 20
habitat_size = 400.0
branching_diffusion = spatialsfs.simulate_branching_diffusion(
    num_steps, s, ndim, d, seed2
)
intensities, weights = spatialsfs.sample(
    branching_diffusion, num_samples, concentration, habitat_size, seed1
)

num_plotted = 0
plt.figure(figsize=(2, 10))
for bd in branching_diffusion.separate_restarts():
    if len(bd) >= 1 / s:
        plt.subplot(5, 1, num_plotted + 1)
        plot_branching_diffusion_2D(bd)
        num_plotted += 1
    if num_plotted >= 5:
        break

# %%prun
spatialsfs.simulate_branching_diffusion(num_steps, s, ndim, d, seed)

# %%prun
spatialsfs.sample(branching_diffusion, num_samples, concentration, habitat_size, seed)


plt.hist(intensities[:, 0], weights=weights, bins=np.arange(0, 0.1, 0.001), log=True)

for concentration in [5, 10, 20, 40]:
    intensities, weights = spatialsfs.sample(
        branching_diffusion, num_samples, concentration, habitat_size, seed1
    )
    plt.hist(
        intensities[:, 0],
        weights=weights,
        bins=np.logspace(-3, 0, 100),
        log=True,
        histtype="step",
        label=concentration,
        density=True,
    )
plt.xscale("log")
plt.legend()

concentration = 20
for separation in np.arange(0, 5, 1):
    intensities, weights = spatialsfs.sample(
        branching_diffusion,
        num_samples,
        concentration,
        habitat_size,
        seed1,
        num_centers=2,
        separation=separation,
    )
    plt.hist2d(
        intensities[:, 0],
        intensities[:, 1],
        weights=weights,
        bins=np.logspace(-5, -1, 100),
        norm=LogNorm(),
    )
    plt.xscale("log")
    plt.yscale("log")
    plt.colorbar()
    plt.show()
