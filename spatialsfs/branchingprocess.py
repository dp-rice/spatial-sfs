"""Simulating and manipulating branching processes."""
from typing import Any, Iterator, Tuple, Union

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
        self.selection_coefficient = selection_coefficient

    def num_restarts(self) -> int:
        """Return the number times the branching process went extinct and restarted."""
        return np.count_nonzero(self.parents == 0) - 1

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


def branch(
    max_steps: int, selection_coefficient: float, seed: int, restart: bool = True
) -> BranchingProcess:
    """Simulate a branching process.

    Parameters
    ----------
    max_steps : int
        The maximum number of steps to simulate.
    selection_coefficient : float
        The selection coefficient against the process. Must be > 0 and < 1.
    seed : int
        A seed for numpy.random random number generation.
    restart : bool
        If True, restart the branching process at the root whenever it goes extinct.
        (Default: True)

    Returns
    -------
    BranchingProcess

    """
    if selection_coefficient <= 0 or selection_coefficient >= 1:
        raise ValueError("selection_coefficient must be > 0 and < 1.")
    return BranchingProcess(None, None, None, selection_coefficient)
