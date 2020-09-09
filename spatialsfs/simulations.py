"""Helper functions for simulations."""
from typing import List, Optional, Tuple

import numpy as np


def simulate_tree(
    selection_coefficient: float, max_steps: int, rng: np.random._generator.Generator
) -> Tuple[List[Optional[int]], np.ndarray, np.ndarray, int]:
    """Simulate parents and birth and death times.

    Parameters
    ----------
    selection_coefficient : float
        The selection coefficient of the branching process.
    max_steps : int
        The maximum number of steps to run the process.
        If not extinct at the end, some individuals will have death time `np.nan`
    rng : np.random._generator.Generator
        A numpy random generator instance.

    Returns
    -------
    parents : List[Optional[int]]
        A list containing the parents.
        `parents[i]` is the index of the parent of individual `i`.
        `parents[i]` is None for the root individual. (Usually only `i==0`)
    birth_times : np.ndarray
        1D array containing the birth times.
    death_times : np.ndarray
        1D array containing the death times.
    num_max : int
        The maximum number of individuals alive at one time.

    """
    # Accumulators to return
    parents: List[Optional[int]] = [None]
    birth_times = [0.0]
    death_times = [np.nan]
    num_max = 1
    # Simulation variables
    time = 0.0
    alive = [0]
    num_total = 1
    for i in range(max_steps):
        time_interval, p, num_offspring = _step(alive, selection_coefficient, rng)
        # Update simulation variables
        time += time_interval
        alive.remove(p)
        alive += [x for x in range(num_total, num_total + num_offspring)]
        num_total += num_offspring
        # Update accumulators
        parents += [p] * num_offspring
        birth_times += [time] * num_offspring
        death_times[p] = time
        death_times += [np.nan] * num_offspring
        num_max = max(num_max, len(alive))
        if len(alive) == 0:
            break
    return parents, np.array(birth_times), np.array(death_times), num_max


def _step(
    alive: List[int], selection_coefficient: float, rng: np.random._generator.Generator
) -> Tuple[float, int, int]:
    n_alive = len(alive)
    time_interval = rng.standard_exponential() / n_alive
    parent = alive[rng.integers(n_alive)]
    num_offspring = 2 * rng.binomial(1, (1 - selection_coefficient) / 2)
    return time_interval, parent, num_offspring


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
