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

# +
from montecarloop import Dealer, JsonFileCache
from spatialsfs.simestimators import DiffusionEstimator

Dealer(JsonFileCache("sim/diffusion.json")).summary(DiffusionEstimator.theory)
# -

# ## Recompute some simulation estimates
#
# In other words, file cache not used.

# +
from montecarloop import Dealer, FakeCache

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

dealer.summary(DiffusionEstimator.theory)
