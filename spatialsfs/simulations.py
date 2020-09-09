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
        selection_coefficient
    max_steps : int
        max_steps
    rng : np.random._generator.Generator
        rng

    Returns
    -------
    Tuple[List[Optional[int]], np.ndarray, np.ndarray, int]

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
    num_offspring = rng.choice(
        [0, 2], p=[(1 + selection_coefficient) / 2, (1 - selection_coefficient) / 2]
    )
    return time_interval, parent, num_offspring


def brownian_bridge(
    t: float,
    t_a: np.array,
    t_b: np.array,
    x_a: np.array,
    x_b: np.array,
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
    rng : np.random._generator.Generator
        A numpy random generator instance.

    Returns
    -------
    np.array
        The positions at t. Shape is (n, ndims).
    """
    means = x_a + (x_b - x_a) * ((t - t_a) / (t_b - t_a))[:, None]
    variances = (t_b - t) * (t - t_a) / (t_b - t_a)
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
        diffusion_coefficient
    ndims: int
        The number of spatial dimensions of the position.
    parents : List[Optional[int]]
        parents
    lifespans : np.ndarray
        lifespans
    rng : np.random._generator.Generator
        rng

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]

    """
    pass
