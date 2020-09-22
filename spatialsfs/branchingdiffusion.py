"""Simulating and analyzing branching diffusions."""

from typing import Iterator, Tuple, Union

import numpy as np

from spatialsfs.branchingprocess import BranchingProcess


class BranchingDiffusion:
    """Branching diffusion simulation output object."""

    def __init__(
        self,
        branching_process: BranchingProcess,
        birth_positions: np.ndarray,
        death_positions: np.ndarray,
        diffusion_coefficient: float,
    ) -> None:
        if birth_positions.dtype != float or death_positions.dtype != float:
            raise TypeError("Position arrays must have dtype float.")
        if (
            birth_positions.ndim != 2
            or death_positions.ndim != 2
            or len(branching_process) != birth_positions.shape[0]
            or len(branching_process) != death_positions.shape[0]
        ):
            raise ValueError(
                (
                    "Position arrays must have shape (n, ndim)"
                    "where n is len(branching_process)."
                )
            )
        self.branching_process = branching_process
        self.birth_positions = birth_positions
        self.death_positions = death_positions
        self.diffusion_coefficient = diffusion_coefficient
        self.ndim: int = birth_positions.shape[1]

    def __eq__(self, other) -> bool:
        """Return True if all attributes are equal elementwise."""
        if isinstance(other, self.__class__):
            return all(
                [
                    self.branching_process == other.branching_process,
                    np.array_equal(
                        self.birth_positions, other.birth_positions, equal_nan=True
                    ),
                    np.array_equal(
                        self.death_positions, other.death_positions, equal_nan=True
                    ),
                    self.diffusion_coefficient == other.diffusion_coefficient,
                ]
            )
        else:
            return NotImplemented

    def __len__(self) -> int:
        """Return the length of BranchingDiffusion: the number of individuals in it."""
        return len(self.branching_process)

    def __getitem__(
        self, idx: Union[int, slice]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Get birth_times, death_times, birth_positions, and death_positions at idx."""
        return (
            self.branching_process.birth_times[idx],
            self.branching_process.death_times[idx],
            self.birth_positions[idx],
            self.death_positions[idx],
        )

    def scale(self) -> float:
        """Return the characteristic spatial scale sqrt(D/s)."""
        return np.sqrt(
            self.diffusion_coefficient / self.branching_process.selection_coefficient
        )

    def num_restarts(self) -> int:
        """Return the number times the branching process went extinct and restarted."""
        return self.branching_process.num_restarts()

    def alive_at(self, time: float) -> np.ndarray:
        """Return an array of bools that are True for individuals alive at `time`."""
        return self.branching_process.alive_at(time)

    def num_alive_at(self, time: float) -> int:
        """Return the number of individuals alive at a time.

        Parameters
        ----------
        time : float
            Time at which to count the living individuals.

        Returns
        -------
        int
            The number of living individuals at `time`.
            Includes individuals moment they are born,
            but not at the moment they die.

        """
        return self.branching_process.num_alive_at(time)

    def separate_restarts(self) -> Iterator["BranchingDiffusion"]:
        """Get each restarted branching process as a separate instance.

        Returns
        -------
        Iterator[BranchingDiffusion]
            The separated branching diffusions.

        """
        restarts = (self.branching_process.parents == 0).nonzero()[0]
        d = self.diffusion_coefficient
        bps = self.branching_process.separate_restarts()
        for start, stop, bp in zip(restarts[1:-1], restarts[2:], bps):
            yield BranchingDiffusion(
                bp,
                _addroot(self.birth_positions[start:stop]),
                _addroot(self.death_positions[start:stop]),
                d,
            )
        yield BranchingDiffusion(
            next(bps),
            _addroot(self.birth_positions[restarts[-1] :]),
            _addroot(self.death_positions[restarts[-1] :]),
            d,
        )


def _addroot(array: np.ndarray) -> np.ndarray:
    new_array = np.zeros((array.shape[0] + 1, array.shape[1]))
    new_array[1:] = array
    return new_array
