"""Simulate branching processes and branching diffusions."""
from typing import List, Tuple

import numpy as np

from spatialsfs import _random
from spatialsfs.branchingdiffusion import BranchingDiffusion
from spatialsfs.branchingprocess import BranchingProcess


def simulate_branching_diffusion(
    num_steps: int,
    selection_coefficient: float,
    ndim: int,
    diffusion_coefficient: float,
    seed: int,
) -> BranchingDiffusion:
    """Simulate a branching diffusion.

    Parameters
    ----------
    num_steps : int
        The number of steps to simulate.
    selection_coefficient : float
        The selection coefficient against the process. Must be > 0 and < 1.
    ndim : int
        The number of spatial dimensions.
    diffusion_coefficient : float
        The diffusion coefficient of the motion.
    seed : int
        A seed for numpy.random random number generation.

    Returns
    -------
    BranchingDiffusion

    """
    seedseq1, seedseq2 = np.random.SeedSequence(seed).spawn(2)
    return diffuse(
        branch(num_steps, selection_coefficient, seedseq1),
        ndim,
        diffusion_coefficient,
        seedseq2,
    )


def branch(
    num_steps: int, selection_coefficient: float, seedseq: np.random.SeedSequence
) -> BranchingProcess:
    """Simulate a branching process.

    Parameters
    ----------
    num_steps : int
        The number of steps to simulate.
    selection_coefficient : float
        The selection coefficient against the process. Must be > 0 and < 1.
    seedseq : np.random.SeedSequence
        A SeedSequence for numpy.random random number generation.

    Returns
    -------
    BranchingProcess

    """
    if selection_coefficient <= 0 or selection_coefficient >= 1:
        raise ValueError("selection_coefficient must be > 0 and < 1.")
    seedseq1, seedseq2, seedseq3 = seedseq.spawn(3)
    return BranchingProcess(
        *_generate_tree(
            _random.raw_times(num_steps, seedseq1),
            _random.num_offspring(num_steps, selection_coefficient, seedseq2),
            _random.parent_choices(num_steps, seedseq3),
        ),
        selection_coefficient,
    )


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
            parents.append(parent)
            birth_times.append(t)
            death_times.append(np.nan)
            num_total += 1
            alive.append(num_total)
    # Everyone dies in the end
    for indiv in alive:
        death_times[indiv] = t
    return np.array(parents), np.array(birth_times), np.array(death_times)


def diffuse(
    branching_process: BranchingProcess,
    ndim: int,
    diffusion_coefficient: float,
    seedseq: np.random.SeedSequence,
) -> BranchingDiffusion:
    """Simulate the positions of a branching diffusion.

    Parameters
    ----------
    branching_process : BranchingProcess
        The branching process defining lifetimes and parental relationships.
    ndim : int
        The number of spatial dimensions.
    diffusion_coefficient : float
        The diffusion coefficient of the motion.
    seedseq : np.random.SeedSequence
        A SeedSequence for numpy.random random number generation.

    Returns
    -------
    BranchingDiffusion

    """
    if ndim <= 0:
        raise ValueError("ndim must be > 0.")
    if diffusion_coefficient <= 0:
        raise ValueError("diffusion_coefficient must be > 0.")
    return BranchingDiffusion(
        branching_process,
        *_generate_positions(
            branching_process,
            _random.raw_distances(len(branching_process), ndim, seedseq),
            diffusion_coefficient,
        ),
        diffusion_coefficient,
    )


def _generate_positions(
    branching_process: BranchingProcess,
    raw_distances: np.ndarray,
    diffusion_coefficient: float,
) -> Tuple[np.ndarray, np.ndarray]:
    assert raw_distances.ndim == 2
    assert raw_distances.shape[0] == len(branching_process)
    assert raw_distances.dtype == float
    birth_positions = np.zeros_like(raw_distances)
    death_positions = np.zeros_like(raw_distances)
    distances = (
        np.sqrt(
            diffusion_coefficient
            * (branching_process.death_times - branching_process.birth_times)
        )[:, None]
        * raw_distances
    )
    for i in range(len(branching_process)):
        birth_positions[i] = death_positions[branching_process.parents[i]]
        death_positions[i] = birth_positions[i] + distances[i]
    return birth_positions, death_positions
