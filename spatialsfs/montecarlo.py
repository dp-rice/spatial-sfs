import math, time, json
from dataclasses import dataclass
from typing import Dict, Iterable, Tuple
from collections.abc import Sequence
from numpy import random
from numpy.random import PCG64
from pathlib import Path
from copy import deepcopy


@dataclass
class SumDuo:
    sum : float = 0
    sumsq : float = 0

    def add(self, value):
        self.sum += value
        self.sumsq += value * value

    def to_hex(self) -> Tuple[str, str]:
        return (float.hex(self.sum), float.hex(self.sumsq))

    @classmethod
    def from_hex(self, pair : Sequence[str]):
        return SumDuo(float.fromhex(pair[0]), float.fromhex(pair[1]))


class SimOutput:

    def __init__(self):
        self.num = 0
        self._sums = None
        self._rng = None
        self._simulator = None

    def restore(self, num, sums: Dict[str, SumDuo]) -> None:
        self.num = num;
        self._sums = deepcopy(sums)

    def copy_sums(self) -> Dict[str, SumDuo]:
        return deepcopy(self._sums)

    def stat_names(self) -> Iterable[str]:
        return self._sums.keys()
        
    def mean(self, name : str) -> float:
        return self._sums[name].sum / self.num

    def std_err(self, name : str) -> float:
        var = self._sums[name].sumsq / self.num - self.mean(name)**2
        var *= self.num / (self.num - 1)
        return math.sqrt(var / self.num)

    def enable_simulation(self, seed_seed, simulator):
        assert self._simulator is None
        self._rng = random.PCG64(seed_seed)
        self._rng.advance(self.num)
        self._simulator = simulator

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


class SimLine:

    def __init__(self, sim_params, seed_seed):
        self.sim_params = sim_params
        self.seed_seed = seed_seed
        self.output = SimOutput()

    def enable_simulation(self, simulator_factory):
        simulator = simulator_factory(self.sim_params)
        self.output.enable_simulation(self.seed_seed, simulator)


class Dealer:

    time_func = time.time
    keyboard_interrupt = False

    def __init__(self, cache, simulator_factory=None):
        self.cache = cache
        self.lines = cache.read()
        if simulator_factory is not None:
            for series in self.lines:
                series.enable_simulation(simulator_factory)

    def output(self, i):
        return self.lines[i].output

    def run(self, exit_after_seconds=1, write_every_seconds=3) -> None:
        now = Dealer.time_func()
        exit_time = now + exit_after_seconds
        while now < exit_time and not Dealer.keyboard_interrupt:
            write_time = min(now + write_every_seconds, exit_time)
            while now < write_time and not Dealer.keyboard_interrupt:
                for line in self.lines:
                    if Dealer.keyboard_interrupt:
                        break
                    line.output.add_simulation()
                now = Dealer.time_func()
            self.cache.write(self.lines) 

    @staticmethod
    def handle_keyboard_interrupts():
        import signal
        signal.signal(signal.SIGINT, Dealer._sigint_handler)
        print("First CTRL-C gracefully waits to save calculation in progress and then exit.")
        print("Second CTRL-C will signal hard keyboard interrupt.")

    @staticmethod
    def _sigint_handler(signum, frame):
        if Dealer.keyboard_interrupt:
            raise KeyboardInterrupt
        print("Waiting for calculation to fininsh before saving and exiting...")
        Dealer.keyboard_interrupt = True


@dataclass
class FakeCache:
    sim_params_by_seed : dict

    def read(self) -> Iterable[SimLine]:
        return [SimLine(p, i) for i, p in self.sim_params_by_seed.items()]

    def write(self, lines: Iterable[SimLine]):
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
                    line.output.restore(d['num'], sums)
                ret.append(line)
        return ret

    def write(self, lines: Iterable[SimLine]):
        pod = []
        for line in lines:
            sums = line.output.copy_sums()
            for s in sums:
                sums[s] = sums[s].to_hex()
            pod.append({
                "sim_params": line.sim_params,
                "seed_seed": line.seed_seed,
                "num": line.output.num,
                "sums": sums
            })
        with open(self.filepath, 'w') as f:
            json.dump(pod, f, indent=2)

