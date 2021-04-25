# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.10.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import math

import numpy as np
from scipy.special import spence

import spatialsfs
from spatialsfs.montecarlo import Dealer, JsonFileCache


# +
def alpha(s):
    b = 0.5 * (1 - s)
    return b / (1 - b)


def theoretical_stats(stat_name, sim_params):
    s = sim_params["s"]
    if stat_name == "ave_alive":
        return (1 + s) / s / 2.0
    a = alpha(s)
    ave_time = -math.log(1 - a) * (1 - a) / a
    if stat_name == "ave_time":
        return ave_time
    if stat_name == "ave_alive_ctime":
        return 1 / ave_time
    if stat_name == "var_time":
        return 2 * spence(1 - a) * (1 - a) / a - ave_time ** 2
    else:
        return np.nan


dealer = Dealer(JsonFileCache("sim/branching.json"))
summary = dealer.summary(theoretical_stats)
# summary.query("stat == 'ave_alive_ctime'")
summary  # [summary['param:s'] <= 0.04]
# -
# # Math Appendix
#
# ## Restarting continuous-time binary branching process
#
# The following stochastic process $Z_t$ should equal the number of survivors in simulations at each time.
#
# Given selection coefficient $s$, let
# $$
# b = \frac{1 - s}{2}
# $$
#
# Let the sequence $\{t_i\}_{i \in \mathbb{Z}}$ be the random branching/death times.
#
# \begin{eqnarray}
#     P(Z_{t_{i+1}} & = & 2 \mid Z_{t_i} = 1) = b   \\
#     P(Z_{t_{i+1}} & = & 1 \mid Z_{t_i} = 1) = 1-b
# \end{eqnarray}
#
# For $n \ge 2$,
# \begin{eqnarray}
#     P(Z_{t_{i+1}} & = & n+1 \mid Z_{t_i} = n) = b   \\
#     P(Z_{t_{i+1}} & = & n-1 \mid Z_{t_i} = n) = 1-b
# \end{eqnarray}
#
# For $n \le 0$ and all $t$,
# $$
# P(Z_t = n) = 0
# $$

# ## Stationary distribution
#
# For any $t$ let
# $$
# q_k = P(Z_t = k)
# $$
#
# From the conditional probabilities of the binary branching process we get:
# $$
# q_1 = (1-b)(q_1 + q_2)
# $$
# and for $k \ge 2$:
# $$
# q_k = b q_{k-1} + (1-b) q_{k+1}
# $$
# and for $k \le 0$:
# $$
# q_k = 0
# $$
#
# Thus
# $$
# b q_1 = (1-b) q_2
# $$
# and for $k \ge 2$:
# $$
# b q_{k-1} = (1-b) q_k  \implies b q_k = (1-b) q_{k+1}
# $$
#
# Thus for all $k \ge 1$:
# $$
# q_{k+1} = \frac{b}{1-b} q_k
# $$
#
# This stationary condition is statisfied by
# $$
# q_k = (1-\alpha) \alpha^{k-1}
# $$
# where $\alpha = \frac{b}{1-b}$ and $k \ge 1$.

# ## Expectation of stationary distribution
#
# Let $T_1$ be the random variable equal to the first positive time of a 'life event' (birth or death) of the Continuous State Branching Process. Let $\{T_i\}_{i \in \mathbb{Z}}$ be the full sequence of times of 'life events' in order.
#
# For any $i \in \mathbb{Z}$,
# \begin{eqnarray}
# E[Z_{T_i}]
# & = & \sum_{k=1}^{\infty} k q_k  \\
# & = & \sum_{k=1}^{\infty} \sum_{j=k}^{\infty} q_j  \\
# & = & \sum_{k=1}^{\infty} \alpha^{k-1} 1  \\
# & = & \frac{1}{1 - \alpha}  \\
# & = & \frac{1-b}{1-2b}  \\
# & = & \frac{(1+s)/2}{s}
# \end{eqnarray}

# ## First and second moments of life-event interval duration
#
# Let $S_{t_i}$ denote the random variable of time until next life-event after live-event at time $t_i$.
#
# \begin{eqnarray*}
# E[S_{t_i}]
#   & = & \sum_{k=1}^\infty q_k \int_0^\infty t k e^{-kt} dt  \\
#   & = & \sum_{k=1}^\infty q_k \frac{1}{k}   \\
#   & = & \frac{1-\alpha}{\alpha} \sum_{k=1}^\infty \frac{\alpha^k}{k}  \\
#   & = & - \frac{1-\alpha}{\alpha} \ln(1-\alpha)
# \end{eqnarray*}
#
# \begin{eqnarray*}
# E[S_{t_i}^2]
#   & = & \sum_{k=1}^\infty q_k \int_0^\infty t^2 k e^{-kt} dt  \\
#   & = & \sum_{k=1}^\infty q_k \frac{2}{k^2}   \\
#   & = & \frac{1-\alpha}{\alpha} \sum_{k=1}^\infty \frac{\alpha^k}{k^2}  \\
#   & = & \frac{1-\alpha}{\alpha} \mathrm{Li}_2(\alpha)
# \end{eqnarray*}
# where $\mathrm{Li}_n$ is the polylogarithm function. The `scipy` package implements `spence` as $\mathrm{Li}_2(1-x)$.

# ## Expected at any point in time
#
# Re-weight conditional probability $q_k$ by expected time duration $1/k$ at level $k$
# \begin{eqnarray*}
# q'_k
#   & = & \frac{ q_k / k }{ \sum_{i=1}^\infty q_i / i }  \\
#   & = & \frac{ q_k } { k E[S_{t_i}] }
# \end{eqnarray*}
# and use it to calculate the expectation at any point in time (not just life-events):
#
# \begin{eqnarray*}
# E[Z_t]
#   & = & \sum_{k=1}^\infty k q'_k  \\
#   & = & \sum_{k=1}^\infty \frac{q_k}{E[S_{t_i}]}  \\
#   & = & \frac{1}{E[S_{t_i}]}  \\
# \end{eqnarray*}
#
#
