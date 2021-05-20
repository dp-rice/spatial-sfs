#!/usr/bin/python3

from pathlib import Path

from montecarloop import Dealer, JsonFileCache
from spatialsfs.simestimators import BranchingEstimator

if __name__ == "__main__":
    dest = Path(__file__).with_name("branching.json")
    seconds = 60 * 60 * 12
    Dealer.handle_keyboard_interrupts()
    Dealer(JsonFileCache(dest), BranchingEstimator).run(seconds)
    print("Done.")
