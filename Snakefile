import gzip, pickle
import numpy as np
from src.simulations import simulate

npops = 10000
s_list = [f'{s:.3f}' for s in np.arange(0.025, 0.251, 0.025)]

rule all:
    input:
        expand('simulations/branchdiff-s={s}.pkl.gz', s=s_list)

rule branchdiff:
    output:
        'simulations/branchdiff-s={s}.pkl.gz'
    run:
        sims = [simulate(s=float(wildcards.s), max_steps=10000) for i in range(npops)]
        with gzip.open(output[0], 'wb') as f:
            f.write(pickle.dumps(sims))
