#!/usr/bin/python3

import spatialsfs
import numpy as np
from spatialsfs.montecarlo import Dealer, JsonFileCache
from pathlib import Path


class BranchingProcessEstimator:

    def __init__(self, sim_params):
        self.num_steps = int(sim_params['num_steps'])
        self.s = sim_params['s']

    def simulate(self, seed):
        branchy = spatialsfs.simulations.branch(self.num_steps, self.s, seed)
        return dict(
            ave_alive=np.mean(branchy.num_alive()),
        )

if __name__ == '__main__':
    Dealer.handle_keyboard_interrupts()
    cache = JsonFileCache(Path(__file__).with_name('branching.json'))
    dealer = Dealer(cache, BranchingProcessEstimator)
    dealer.run(60*60)
    print("Done.")

