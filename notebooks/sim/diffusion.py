#!/usr/bin/python3

from pathlib import Path

from spatialsfs.montecarlo import Dealer, JsonFileCache
from spatialsfs.simestimators import DiffusionEstimator

if __name__ == "__main__":
    dest = Path(__file__).with_name("diffusion.json")
    seconds = 60 * 60 * 12
    Dealer.handle_keyboard_interrupts()
    Dealer(JsonFileCache(dest), DiffusionEstimator).run(seconds)
    print("Done.")
