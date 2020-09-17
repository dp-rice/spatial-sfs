"""Helper functions for simulations."""
from typing import List, Optional, Tuple

import numpy as np

from spatialsfs.branchingprocess import BranchingProcess


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
            death_times.append(np.inf)
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
            death_times.append(np.inf)
    return np.array(parents), np.array(birth_times), np.array(death_times)


def brownian_bridge(
    t: float,
    t_a: np.array,
    t_b: np.array,
    x_a: np.array,
    x_b: np.array,
    diffusion_coefficient: float,
    rng: np.random._generator.Generator,
) -> np.array:
    """Return random positions drawn from n independent Brownian bridges.

    Parameters
    ----------
    t : float
        The time at which to sample the Brownian bridges.
    t_a : np.array
        1D array with the initial times of the bridges.
        Shape is (n,)
    t_b : np.array
        1D array with the final times of the bridges.
        Shape is (n,)
    x_a : np.array
        2D array with the initial positions of the bridges.
        Shape is (n, ndims)
    x_b : np.array
        2D array with the initial positions of the bridges.
        Shape is (n, ndims)
    diffusion_coefficient : float
        The diffusion coefficient of the brownian bridge.
    rng : np.random._generator.Generator
        A numpy random generator instance.

    Returns
    -------
    np.array
        The positions at t. Shape is (n, ndims).
    """
    means = x_a + (x_b - x_a) * ((t - t_a) / (t_b - t_a))[:, None]
    variances = diffusion_coefficient * (t_b - t) * (t - t_a) / (t_b - t_a)
    return means + np.sqrt(variances)[:, None] * rng.standard_normal(size=x_a.shape)


def simulate_positions(
    diffusion_coefficient: float,
    ndims: int,
    parents: List[Optional[int]],
    lifespans: np.ndarray,
    rng: np.random._generator.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate positions, given parental relationships and birth/death times.

    Parameters
    ----------
    diffusion_coefficient : float
        The diffusion coefficient of the process.
    ndims: int
        The number of spatial dimensions of the position.
    parents : List[Optional[int]]
        A list containing the parents.
        `parents[i]` is the index of the parent of individual `i`.
        `parents[i]` is None for the root individual. (Usually only `i==0`)
    lifespans : np.ndarray
        1D array of lifespans of individuals.
    rng : np.random._generator.Generator
        A numpy random generator instance.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        2D arrays containing the birth and death positions.
        Both arrays have shape `(n, ndims)` where n is the number of individuals.

    """
    if ndims < 1:
        raise ValueError("ndims must be >= 1 to simulate positions.")
    n_indiv = len(parents)
    birth_positions = np.empty((n_indiv, ndims), dtype=float)
    death_positions = np.empty((n_indiv, ndims), dtype=float)
    scales = np.sqrt(diffusion_coefficient * lifespans)
    distances_traveled = scales[:, None] * rng.standard_normal(size=(n_indiv, ndims))
    for i in range(n_indiv):
        if parents[i] is None:
            birth_positions[i] = 0.0
        else:
            birth_positions[i] = death_positions[parents[i]]
        death_positions[i] = birth_positions[i] + distances_traveled[i]
    return birth_positions, death_positions
