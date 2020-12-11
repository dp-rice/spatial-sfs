"""Run simulations over parameter grid. WARNING: Code is of date."""
import gzip, pickle
import numpy as np
from spatialsfs.simulations import simulate, save_populations

npops = 10000
s_list = [f'{s:.3f}' for s in np.arange(0.02, 0.05, 0.01)]
s_list += [f'{s:.3f}' for s in np.arange(0.025, 0.251, 0.025)]

rule all:
    input:
        expand('simulations/branchdiff-s={s}.pkl.gz', s=s_list)

# FIXME
rule branchdiff:
    output:
        'simulations/branchdiff-s={s}.pkl.gz'
    run:
        seed = 100
        sims = simulate(npops, seed, s=float(wildcards.s))
        save_populations(sims, output[0])
