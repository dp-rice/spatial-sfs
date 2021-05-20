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

from montecarloop import Dealer, JsonFileCache


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
summary[summary["pvalue"] <= 0.05]  # .query("stat == 'ave_alive'")
# -
# # Math Appendix
#
# Let $\Omega$ denote the probability sample space of paths.
#
# Given any reference time $r$, define the "after duration time" random variable $A_r(\omega) \in (0, \infty)$ to be the duration of time from $r$ until the next life event (birth/death) for sample path $\omega \in \Omega$.
#
# Similarly, define the "before duration time" random variable $B_r(\Omega) \in [0, \infty)$ to be the duration of time from the most recent life event until $r$ for sample path $\omega \in \Omega$

# ## Restarting continuous-time binary branching process
#
# The following stochastic process $Z_t$ should equal the number of survivors in simulations at each time.
#
# Given selection coefficient $s$, let
# $$
# b = \frac{1 - s}{2}
# $$
#
# For any reference time $r$,
#
# \begin{eqnarray}
#     P(Z_{r+A_r} & = & 2 \mid Z_r = 1 \wedge B_r = 0) = b   \\
#     P(Z_{r+A_r} & = & 1 \mid Z_r = 1 \wedge B_r = 0) = 1-b
# \end{eqnarray}
#
# For $n \ge 2$,
# \begin{eqnarray}
#     P(Z_{r+A_r} & = & n+1 \mid Z_r = n \wedge B_r = 0) = b   \\
#     P(Z_{r+A_r} & = & n-1 \mid Z_r = n \wedge B_r = 0) = 1-b
# \end{eqnarray}
#
# For $n \le 0$ and all $t$,
# $$
# P(Z_t = n) = 0
# $$

# ## Stationary distribution
#
# Define
# $$
# q_k = P(Z_r = k \mid B_r = 0)
# $$
#
# where $r$ can be any reference time since $Z_t$ is stationary.
#
# From the conditional probabilities of the binary branching process we get:
#
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
# \begin{eqnarray}
# E[Z_r | B_r = 0]
# & = & \sum_{k=1}^{\infty} k q_k  \\
# & = & \sum_{k=1}^{\infty} \sum_{j=k}^{\infty} q_j  \\
# & = & \sum_{k=1}^{\infty} \alpha^{k-1} 1  \\
# & = & \frac{1}{1 - \alpha}  \\
# & = & \frac{1-b}{1-2b}  \\
# & = & \frac{(1+s)/2}{s}
# \end{eqnarray}

# ## First and second moments of life-event interval duration
#
# By definition of (Markovian) continuous-time branching process with $\lambda=1$,
# \begin{eqnarray*}
# P(A_r \le a | Z_r = k \wedge B_r = b) & = & \int_0^a k e^{-kt} dt
# \end{eqnarray*}
# for any $b \in [0, \infty)$. Thus
# \begin{eqnarray*}
# E[A_r | B_r = 0]
#   & = & \sum_{k=1}^\infty q_k \int_0^\infty t k e^{-kt} dt  \\
#   & = & \sum_{k=1}^\infty q_k \frac{1}{k}   \\
#   & = & \frac{1-\alpha}{\alpha} \sum_{k=1}^\infty \frac{\alpha^k}{k}  \\
#   & = & - \frac{1-\alpha}{\alpha} \ln(1-\alpha)
# \end{eqnarray*}
#
# \begin{eqnarray*}
# E[A_{t_i}^2 | B_r = 0]
#   & = & \sum_{k=1}^\infty q_k \int_0^\infty t^2 k e^{-kt} dt  \\
#   & = & \sum_{k=1}^\infty q_k \frac{2}{k^2}   \\
#   & = & \frac{1-\alpha}{\alpha} \sum_{k=1}^\infty \frac{\alpha^k}{k^2}  \\
#   & = & \frac{1-\alpha}{\alpha} \mathrm{Li}_2(\alpha)
# \end{eqnarray*}
# where $\mathrm{Li}_n$ is the polylogarithm function. The `scipy` package implements `spence` as $\mathrm{Li}_2(1-x)$.

# ## Intuitive calculation of continuous-time expectation
#
# Re-weight conditional probability $q_k$ by expected time duration $1/k$ at level $k$
# \begin{eqnarray*}
# q'_k
#   & = & \frac{ q_k / k }{ \sum_{i=1}^\infty q_i / i }  \\
#   & = & \frac{ q_k } { k E[A_r | B_r=0] }
# \end{eqnarray*}
# and use it to calculate the expectation at any point in time (not just life-events):
#
# \begin{eqnarray*}
# E[Z_t]
#   & = & \sum_{k=1}^\infty k q'_k  \\
#   & = & \sum_{k=1}^\infty \frac{q_k}{E[A_r|B_r=0]}  \\
#   & = & \frac{1}{E[A_r|B_r=0]}  \\
# \end{eqnarray*}
#
#

# ## Formal proof of continuous-time expectation
#
# For any real $t$, let $m_t$ map all sample paths $\omega \in \Omega$ to paths "shifted" in time by $t$ (forward if $t$ positive, backward if $t$ negative). Since the process is stationary, this mapping $m_t$ is probability distribution preserving.
#
# By construction, for every reference time $r$ and time shift $t \ge 0$,
# \begin{eqnarray*}
# \{ m_{-t}(\omega) : A_r(\omega) > t \wedge Z_r(\omega) = k \}
#   & = & \{ \omega : B_r(\omega) > t \wedge Z_r(\omega) = k \}
# \end{eqnarray*}
# and thus for all $t$,
# \begin{eqnarray*}
# P( A_r > t \mid Z_r = k ) & = & P( B_r > t \mid Z_r = k)
# \end{eqnarray*}
#
# and thus similar to $P( A_r \le a \mid Z_r = k)$,
# \begin{eqnarray*}
# P( B_r \le b \mid Z_r = k) & = & \int_0^b k e^{-kt} dt
# \end{eqnarray*}
# for any $b \ge 0$.
#
# Let $f(t)$ denote the density function of $B_r$,
# \begin{eqnarray*}
# P(B_r \le b)
#   & = & \sum_{k=1}^\infty P( B_r \le b \mid Z_r = k) P(Z_r = k)  \\
#   & = & \sum_{k=1}^\infty \int_0^b k e^{-kt} dt P(Z_r = k)  \\
#   & = & \int_0^b \sum_{k=1}^\infty P(Z_r = k) k e^{-kt} dt
# \end{eqnarray*}
# thus we have density function of $B_r$ as
# \begin{eqnarray*}
# f(t) & := & \sum_{k=1}^\infty P(Z_r = k) k e^{-kt}
# \end{eqnarray*}
# In particular, $f(0) = E[Z_r]$.
#
# \begin{eqnarray*}
# P(Z_r = k \wedge B_r \le b) & = & P(Z_0 = k) P(B_r \le b \mid Z_r = k)  \\
# \int_0^b P(Z_r = k \mid B_r = t) f(t) dt & = & \int_0^b P(Z_0 = k) k e^{-kt} dt \\
# P(Z_r = k \mid B_r = t) f(t) & = & P(Z_0 = k) k e^{-kt}  \\
# \end{eqnarray*}
#
# Evaluating at $t=0$ results in
# \begin{eqnarray*}
# q_k E[Z_r] & = & P(Z_0 = k) k  \\
# \frac{q_k}{k} E[Z_r] & = & P(Z_0 = k)  \\
# \sum_{k=1}^\infty \frac{q_k}{k} E[Z_r] & = & 1  \\
# \end{eqnarray*}
#
# Making use of previous results for $E[A_r | B_r = 0]$,
#
# \begin{eqnarray*}
# E[A_r | B_r = 0] E[Z_r] & = & 1  \\
# E[Z_r] & = & \frac{-\alpha}{(1-\alpha) \ln(1-\alpha)}
# \end{eqnarray*}
#
