import math, time, json
from dataclasses import dataclass
from typing import Dict, Iterable, Tuple
from numpy import random
from numpy.random import PCG64
from pathlib import Path


@dataclass
class SumDuo:
    sum : float = 0
    sumsq : float = 0

    def add(self, value):
        self.sum += value
        self.sumsq += value * value

    def to_hex(self) -> Tuple[float, float]:
        return (float.hex(self.sum), float.hex(self.sumsq))

    @classmethod
    def from_hex(self, pair : Iterable[float]):
        return SumDuo(float.fromhex(pair[0]), float.fromhex(pair[1]))


class SimLine:

    def __init__(self, sim_params, seed_seed):
        self.sim_params = sim_params
        self.seed_seed = seed_seed
        self.num = 0
        self._sums = None
        self._rng = None
        self._simulator = None

    def restore(self, num, sums: Dict[str, SumDuo]) -> None:
        self.num = num;
        self._sums = sums.copy()

    def copy_sums(self):
        return self._sums.copy()

    def enable_simulation(self, simulator_factory):
        assert self._simulator is None
        self._rng = random.PCG64(self.seed_seed)
        self._rng.advance(self.num)
        self._simulator = simulator_factory(self.sim_params)

    def add_simulation(self):
        assert self._simulator is not None
        self.num += 1
        seed = self._rng.random_raw()
        stats = self._simulator.simulate(seed)
        if self._sums is None:
            self._sums = {name: SumDuo() for name in stats.keys()}
        else:
            assert stats.keys() == self._sums.keys()
        for name, val in stats.items():
            self._sums[name].add(val)

    def stat_names(self) -> Iterable[str]:
        return self._sums.keys()
        
    def sim_mean(self, name : str) -> float:
        return self._sums[name].sum / self.num

    def sim_std_err(self, name : str) -> float:
        var = self._sums[name].sumsq / self.num - self.sim_mean(name)**2
        var *= self.num / (self.num - 1)
        return math.sqrt(var / self.num)


class Dealer:

    time_func = time.time

    def __init__(self, cache, simulator_factory=None):
        self.cache = cache
        self.tallies = cache.read()
        if simulator_factory is not None:
            for series in self.tallies:
                series.enable_simulation(simulator_factory)

    def run(self, exit_after_seconds=1, write_every_seconds=3) -> None:
        now = Dealer.time_func()
        exit_time = now + exit_after_seconds
        while now < exit_time:
            write_time = min(now + write_every_seconds, exit_time)
            while now < write_time:
                for tally in self.tallies:
                    tally.add_simulation()
                now = Dealer.time_func()
            self.cache.write(self.tallies) 


@dataclass
class FakeCache:
    sim_params_by_seed : dict

    def read(self) -> Iterable[SimLine]:
        return [SimLine(p, i) for i, p in self.sim_params_by_seed.items()]

    def write(self, tallies : Iterable[SimLine]):
        pass


@dataclass
class JsonFileCache:
    filepath : Path

    def read(self) -> Iterable[SimLine]:
        ret = []
        with open(self.filepath) as f:
            for d in json.load(f):
                line = SimLine(d['sim_params'], d['seed_seed'])
                if 'num' in d:
                    sums = d['sums']
                    for s in sums:
                        sums[s] = SumDuo.from_hex(sums[s])
                    line.restore(d['num'], sums)
                ret.append(line)
        return ret

    def write(self, tallies : Iterable[SimLine]):
        pod = []
        for tally in tallies:
            sums = tally.copy_sums()
            for s in sums:
                sums[s] = sums[s].to_hex()
            pod.append({
                "sim_params": tally.sim_params,
                "seed_seed": tally.seed_seed,
                "num": tally.num,
                "sums": sums
            })
        with open(self.filepath, 'w') as f:
            json.dump(pod, f, indent=2)

