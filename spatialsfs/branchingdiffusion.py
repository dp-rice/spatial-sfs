"""Simulating and analyzing branching diffusions."""

from typing import Tuple, Union

import numpy as np

import spatialsfs.simulations as simulations
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

    def positions_at(
        self, time: float, rng: np.random._generator.Generator
    ) -> np.ndarray:
        """Return random interpolated positions of individuals alive at a time.

        Parameters
        ----------
        time : float
            time

        Returns
        -------
        positions : np.ndarray
            1D array of interpolated positions.
            The length of the array is the number of individuals alive at `time`.
            Includes individuals the moment they are born,
            but not at the moment they die.
        rng : np.random._generator.Generator
            A numpy random generator instance.

        Notes
        -----
        The positions are modeled as a Brownian bridge, conditional on birth and death
        times and locations. Repeated calls to `positions_at` will generate independent
        draws from the Brownian bridge.

        """
        if len(self.birth_positions) == 0:
            raise RuntimeError("Positions not simulated.")
        if self.diffusion_coefficient is None:
            raise RuntimeError("Diffusion coefficient not set.")
        return simulations.brownian_bridge(
            time, *self[self.alive_at(time)], self.diffusion_coefficient, rng
        )


# def simulate_branching_diffusions(
#     num_reps: int,
#     selection_coefficient: float,
#     diffusion_coefficient: float = 1.0,
#     ndim: int = 1,
#     max_steps: int = 10000,
#     rng: Optional[np.random._generator.Generator] = None,
# ) -> List[BranchingDiffusion]:
#     """Simulate replicate branching diffusions.

#     Parameters
#     ----------
#     num_reps : int
#         The number of replicate simulations to run.
#     selection_coefficient : float
#         The selection coefficient for all simulations.
#     diffusion_coefficient : float
#         The diffusion coefficient for all simulations.
#     ndim : int
#         The number of spatial dimensions of the position.
#         Default: 1
#     max_steps : int
#         The maximum number of steps to run the simulations for.
#         Default: 10000
#     rng : Optional[np.random._generator.Generator]
#         The numpy random generator to use for rng.
#         Default: create a new genertor with np.random.default_generator()

#     Returns
#     -------
#     List[BranchingDiffusion]

#     """
#     bds = []
#     if rng is None:
#         rng = np.random.default_rng()
#     for i in range(num_reps):
#         bd = BranchingDiffusion()
#         bd.simulate_tree(selection_coefficient, max_steps, rng)
#         bd.simulate_positions(diffusion_coefficient, ndim, rng)
#         bds.append(bd)
#     return bds
