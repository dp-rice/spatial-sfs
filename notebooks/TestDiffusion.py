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


def theoretical_stats(stat_name, sim_params):
    if stat_name == "var_time_adj_dist":
        return sim_params["diffusion"] * sim_params["ndim"]
    else:
        return 0.0


# +
from montecarloop import Dealer, JsonFileCache

Dealer(JsonFileCache("sim/diffusion.json")).summary(theoretical_stats)
# -

# ## Recompute some simulation estimates
#
# In other words, file cache not used.

# +
from montecarloop import Dealer, FakeCache
from spatialsfs.simestimators import DiffusionEstimator

common_params = {"nstep": 100000, "s": 0.2, "ndim": 2}
cache = FakeCache(
    {
        667: dict(diffusion=1.0, **common_params),
        668: dict(diffusion=3.0, **common_params),
    }
)
dealer = Dealer(cache, DiffusionEstimator)
dealer.run(5)
# -

dealer.summary(theoretical_stats)
