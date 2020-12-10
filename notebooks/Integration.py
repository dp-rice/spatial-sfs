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

import spatialsfs.theory as th

# ## Testing the integrator

# Some simple test functions to integrate:

# +
def f(a):
    return np.sum(a, axis=1)


def g(a, b):
    return np.sum(a * b, axis=1)


def h(a):
    return np.sum(a ** 2, axis=1)


def f_prime(a, p=1):
    return np.sum(a ** p, axis=1)


# -

# Regardless of the function to integrate and the number of dimensions,
# the observed absolute error should scale like the standard error
# as we increase the sample size.

for n_samples in np.logspace(1, 5, 20):
    mean, sterr = th.gaussian_integral(f, int(n_samples), 1, 1, 100)
    plt.loglog(
        n_samples, np.abs(mean), "o", color="C0",
    )
    plt.loglog(n_samples, sterr, "o", color="C1")
plt.plot([], [], "o", color="C0", label=r"|error|")
plt.plot([], [], "o", color="C1", label=r"std-error")
plt.legend()

for n_samples in np.logspace(1, 5, 20):
    mean, sterr = th.gaussian_integral(f, int(n_samples), 1, 2, 100)
    plt.loglog(n_samples, np.abs(mean), "o", color="C0")
    plt.loglog(n_samples, sterr, "o", color="C1")
plt.plot([], [], "o", color="C0", label=r"|error|")
plt.plot([], [], "o", color="C1", label=r"std-error")
plt.legend()

for n_samples in np.logspace(1, 5, 20):
    mean, sterr = th.gaussian_integral(g, int(n_samples), 2, 1, 100)
    plt.loglog(n_samples, np.abs(mean), "o", color="C0")
    plt.loglog(n_samples, sterr, "o", color="C1")
plt.plot([], [], "o", color="C0", label=r"|error|")
plt.plot([], [], "o", color="C1", label=r"std-error")
plt.legend()

for n_samples in np.logspace(1, 5, 20):
    mean, sterr = th.gaussian_integral(g, int(n_samples), 2, 2, 100)
    plt.loglog(n_samples, np.abs(mean), "o", color="C0")
    plt.loglog(n_samples, sterr, "o", color="C1")
plt.plot([], [], "o", color="C0", label=r"|error|")
plt.plot([], [], "o", color="C1", label=r"std-error")
plt.legend()

for n_samples in np.logspace(1, 5, 20):
    mean, sterr = th.gaussian_integral(h, int(n_samples), 1, 1, 100)
    plt.loglog(n_samples, np.abs(mean - 1), "o", color="C0")
    plt.loglog(n_samples, sterr, "o", color="C1")
plt.plot([], [], "o", color="C0", label=r"|error|")
plt.plot([], [], "o", color="C1", label=r"std-error")
plt.legend()

for n_samples in np.logspace(1, 5, 20):
    mean, sterr = th.gaussian_integral(h, int(n_samples), 1, 2, 100)
    plt.loglog(n_samples, np.abs(mean - 2), "o", color="C0")
    plt.loglog(n_samples, sterr, "o", color="C1")
plt.plot([], [], "o", color="C0", label=r"|error|")
plt.plot([], [], "o", color="C1", label=r"std-error")
plt.legend()

for n_samples in np.logspace(1, 5, 20):
    mean, sterr = th.gaussian_integral(f_prime, int(n_samples), 1, 1, 100, p=1)
    plt.loglog(n_samples, np.abs(mean), "o", color="C0")
    plt.loglog(n_samples, sterr, "o", color="C1")
plt.plot([], [], "o", color="C0", label=r"|error|")
plt.plot([], [], "o", color="C1", label=r"std-error")
plt.legend()

for n_samples in np.logspace(1, 5, 20):
    mean, sterr = th.gaussian_integral(f_prime, int(n_samples), 1, 1, 100, p=2)
    plt.loglog(n_samples, np.abs(mean - 1), "o", color="C0")
    plt.loglog(n_samples, sterr, "o", color="C1")
plt.plot([], [], "o", color="C0", label=r"|error|")
plt.plot([], [], "o", color="C1", label=r"std-error")
plt.legend()
