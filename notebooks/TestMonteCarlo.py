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

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import pade

rng = np.random.default_rng()


# +
def a2(d, nsamples=10000):
    x = rng.standard_normal(size=nsamples)
    return 2 * np.pi * np.mean(np.exp(-(x ** 2) / 2) / (1 + d * x ** 2))


def a3(d, nsamples=10000):
    x1 = rng.standard_normal(size=nsamples)
    x2 = rng.standard_normal(size=nsamples)
    return (2 * np.pi) ** 2 * np.mean(
        np.exp(-((x1 + x2) ** 2) / 2)
        / ((2 + 2 * d * (x1 + x2) ** 2) * (3 + d * (x1 + x2) ** 2 + x1 ** 2 + x2 ** 2))
    )


# -


def coeffs(d, nsamples=10000):
    p, q = pade([a2(d), a3(d)], 1)
    return p.coefficients[0], q.coefficients[0]


a2(1)

a3(1)

d = 1
p, q = pade([a2(d), a3(d)], 1)
coeffs(d)

q.coefficients[0]

D = np.logspace(-2, 2, 10)
for d in D:
    mut, sel = coeffs(d)
    plt.loglog(d, mut, "o", color="C0")

D = np.logspace(-2, 2, 10)
for d in D:
    mut, sel = coeffs(d)
    plt.loglog(d, -sel, "o", color="C1")
