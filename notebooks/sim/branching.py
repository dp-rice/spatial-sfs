#!/usr/bin/python3

from pathlib import Path

import numpy as np

import spatialsfs
from spatialsfs.montecarlo import Dealer, JsonFileCache


class BranchingProcessEstimator:
    def __init__(self, sim_params):
        self.num_steps = int(sim_params["num_steps"])
        self.omit_steps = int(sim_params.get("omit_steps", 0))
        self.s = sim_params["s"]

    def simulate(self, seed):
        branchy = spatialsfs.simulations.branch(self.num_steps, self.s, seed)
        nums = branchy.num_alive()
        return dict(
            ave_alive=np.mean(nums[self.omit_steps :]), ave_alive_0=np.mean(nums)
        )


if __name__ == "__main__":
    Dealer.handle_keyboard_interrupts()
    cache = JsonFileCache(Path(__file__).with_name("branching.json"))
    dealer = Dealer(cache, BranchingProcessEstimator)
    dealer.run(60 * 60 * 12)
    print("Done.")
