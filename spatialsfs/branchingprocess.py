"""Simulating and manipulating branching processes."""
from typing import Any, Iterator, List, Tuple, Union

import numpy as np


class BranchingProcess:
    """An instance of a branching process."""

    def __init__(
        self,
        parents: np.ndarray,
        birth_times: np.ndarray,
        death_times: np.ndarray,
        selection_coefficient: float,
    ) -> None:
        if parents.dtype != int:
            raise TypeError("parents must be a np.ndarray of ints.")
        if birth_times.dtype != float:
            raise TypeError("birth_times must be a np.ndarray of floats.")
        if death_times.dtype != float:
            raise TypeError("death_times must be a np.ndarray of floats.")
        if parents.shape != birth_times.shape:
            raise ValueError("parents and birth_times must have the same shape.")
        if parents.shape != death_times.shape:
            raise ValueError("parents and death_times must have the same shape.")
        if parents[0] != 0 or birth_times[0] != 0.0 or death_times[0] != 0.0:
            raise ValueError(
                """parents[0], birth_times[0], and death_times[0] must be zero
                (stands for the root of the tree)"""
            )
        self.parents = parents
        self.birth_times = birth_times
        self.death_times = death_times
        self.final_time: float = np.nanmax(death_times)
        self.selection_coefficient = selection_coefficient

    def num_restarts(self) -> int:
        """Return the number times the branching process went extinct and restarted."""
        return np.count_nonzero(self.parents == 0) - 1

    def num_alive_at(self, time: float) -> int:
        """Return the number of individuals alive at time."""
        if time > self.final_time:
            raise ValueError(f"time ({time}) > final_time ({self.final_time})")
        return np.count_nonzero((self.birth_times <= time) & (self.death_times > time))

    def __eq__(self, other: Any) -> bool:
        """Return true if all attributes are equal."""
        if isinstance(other, self.__class__):
            return all(
                [
                    np.array_equal(self.parents, other.parents, equal_nan=True),
                    np.array_equal(self.birth_times, other.birth_times, equal_nan=True),
                    np.array_equal(self.death_times, other.death_times, equal_nan=True),
                    self.selection_coefficient == other.selection_coefficient,
                ]
            )
        else:
            return NotImplemented

    def __getitem__(
        self, idx: Union[int, slice]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get parents, birth_times, death_times at idx."""
        return (self.parents[idx], self.birth_times[idx], self.death_times[idx])

    def __len__(self) -> int:
        """Let `len(bp)` be the length of its arrays."""
        return len(self.parents)


def separate_restarts(bp: BranchingProcess) -> Iterator[BranchingProcess]:
    """Get each restarted branching process as a separate instance.

    Parameters
    ----------
    bp : BranchingProcess
        The BranchingProcess to separate.

    Returns
    -------
    Iterator[BranchingProcess]
        The separated branching processes.

    """
    restarts = (bp.parents == 0).nonzero()[0]
    s = bp.selection_coefficient
    for start, stop in zip(restarts[1:-1], restarts[2:]):
        yield BranchingProcess(*_reroot(*bp[start:stop], start), s)
    yield BranchingProcess(*_reroot(*bp[restarts[-1] :], restarts[-1]), s)


def _reroot(
    parents: np.ndarray, birth_times: np.ndarray, death_times: np.ndarray, old_root: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    new_parents = np.copy(parents)
    new_parents[1:] -= old_root - 1
    return (
        _addroot(new_parents),
        _addroot(birth_times - birth_times[0]),
        _addroot(death_times - birth_times[0]),
    )


def _addroot(array: np.ndarray) -> np.ndarray:
    return np.insert(array, 0, 0)


def branch(num_steps: int, selection_coefficient: float, seed: int) -> BranchingProcess:
    """Simulate a branching process.

    Parameters
    ----------
    num_steps : int
        The number of steps to simulate.
    selection_coefficient : float
        The selection coefficient against the process. Must be > 0 and < 1.
    seed : int
        A seed for numpy.random random number generation.

    Returns
    -------
    BranchingProcess

    """
    if selection_coefficient <= 0 or selection_coefficient >= 1:
        raise ValueError("selection_coefficient must be > 0 and < 1.")
    seed1, seed2, seed3 = np.random.SeedSequence(seed).spawn(3)
    return BranchingProcess(
        *_generate_tree(
            _raw_times(num_steps, seed1),
            _num_offspring(num_steps, selection_coefficient, seed2),
            _parent_choices(num_steps, seed3),
        ),
        selection_coefficient,
    )


def _raw_times(num_steps: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).standard_exponential(size=num_steps)


def _num_offspring(num_steps: int, s: float, seed: int) -> np.ndarray:
    return 2 * np.random.default_rng(seed).binomial(1, (1 - s) / 2, size=num_steps)


def _parent_choices(num_steps: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).random(size=num_steps)


def _generate_tree(
    raw_times: np.ndarray, num_offspring: np.ndarray, parent_choices: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    assert raw_times.shape == num_offspring.shape
    assert raw_times.shape == parent_choices.shape
    assert raw_times.dtype == float
    assert num_offspring.dtype == int
    assert parent_choices.dtype == float
    t = 0.0
    alive: List[int] = []
    num_total = 0
    # Root of the tree
    parents = [0]
    birth_times = [0.0]
    death_times = [0.0]
    for raw_interval, noff, parent_float in zip(
        raw_times, num_offspring, parent_choices
    ):
        if not alive:
            # Restart extinct process.
            parents.append(0)
            birth_times.append(t)
            death_times.append(np.nan)
            num_total += 1
            alive.append(num_total)
        t += raw_interval / len(alive)
        # Kill parent
        parent = alive.pop(int(parent_float * len(alive)))
        death_times[parent] = t
        # Reproduce
        for i in range(noff):
            num_total += 1
            alive.append(num_total)
            parents.append(parent)
            birth_times.append(t)
            death_times.append(np.nan)
    return np.array(parents), np.array(birth_times), np.array(death_times)
