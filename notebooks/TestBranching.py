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

import spatialsfs
from spatialsfs.montecarlo import Dealer, JsonFileCache
import numpy as np

dealer = Dealer(JsonFileCache('sim/branching.json'))
[(line.sim_params['s'], line.output.mean('ave_alive'), line.output.std_err('ave_alive'), line.output.num,)
 for line in dealer.lines]


# +
def theory_mean_alive(s):
    return (1+s)/s/2.0

theory_mean_alive(np.array([0.01, 0.02, 0.04, 0.08, 0.16, 0.32]))
# -



# # Math Appendix

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
# \begin{eqnarray}
# E\left[ Z_t \mid t \in \{t_i\}_{i \in \mathbb{Z}} \right]
# & = & \sum_{k=1}^{\infty} k q_k  \\
# & = & \sum_{k=1}^{\infty} \sum_{j=k}^{\infty} q_j  \\
# & = & \sum_{k=1}^{\infty} \alpha^{k-1} 1  \\
# & = & \frac{1}{1 - \alpha}
# & = & \frac{1-b}{1-2b}
# & = & \frac{(1+s)/2}{s}
# \end{eqnarray}
