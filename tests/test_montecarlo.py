"""Test spatialsfs.montecarlo."""

import math
from pytest import approx
import numpy as np
from spatialsfs.montecarlo import *


def fake_time_func():
    fake_time_func.seconds += 1/2048
    return fake_time_func.seconds

Dealer.time_func = fake_time_func


class PiEstimator:
    def __init__(self, sim_params):
        self.n = int(sim_params['n'])

    def simulate(self, seed):
        rng = np.random.default_rng(seed)
        # generate random (x,y) points in square [-1,1)x[-1,1)
        x = rng.random(self.n) * 2 - 1
        y = rng.random(self.n) * 2 - 1
        inside_circle = (x**2 + y**2 < 1)
        return dict(pi=(4 * np.sum(inside_circle) / self.n))


def assert_pi(tally, expected_num):
    assert tally.stat_names() == set(['pi'])
    assert tally.num == expected_num
    assert tally.sim_mean('pi') == approx(math.pi, 0.0004)
    assert tally.sim_std_err('pi') == approx(0.001, 0.3)


def test_monte_carlo_pi():
    fake_time_func.seconds = 0
    cache = FakeCache({666:dict(n=1000)})
    dealer = Dealer(cache, PiEstimator)
    assert len(dealer.tallies) == 1
    assert dealer.tallies[0].num == 0
    dealer.run()
    assert_pi(dealer.tallies[0], 2048)
    dealer.run()
    assert_pi(dealer.tallies[0], 4096)


def test_monte_carlo_pi_file(tmpdir):
    fake_time_func.seconds = 0
    path = tmpdir.join('test_sim_cache.json')
    contents = """[ {
        "sim_params":{"n": 1000},
        "seed_seed": 666
    } ]"""
    with open(path, 'w') as f:
        f.write(contents)
    dealer = Dealer(JsonFileCache(path), PiEstimator)
    assert dealer.tallies[0].num == 0
    dealer.run()
    assert len(dealer.tallies) == 1
    assert_pi(dealer.tallies[0], 2048)
    pi_estimate = dealer.tallies[0].sim_mean('pi')

    read_only_dealer = Dealer(JsonFileCache(path))
    assert read_only_dealer.tallies[0].sim_mean('pi') == pi_estimate
    assert_pi(dealer.tallies[0], 2048)

    dealer2 = Dealer(JsonFileCache(path), PiEstimator)
    assert len(dealer2.tallies) == 1
    assert_pi(dealer2.tallies[0], 2048)
    dealer2.run()
    assert_pi(dealer2.tallies[0], 4096)

    nofile_dealer = Dealer(FakeCache({666:dict(n=1000)}), PiEstimator)
    nofile_dealer.run()
    nofile_dealer.run()
    assert_pi(nofile_dealer.tallies[0], 4096)
    assert dealer2.tallies[0].sim_mean('pi') == nofile_dealer.tallies[0].sim_mean('pi')

