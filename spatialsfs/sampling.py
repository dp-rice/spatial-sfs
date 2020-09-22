"""Functions for sampling positions and the SFS."""

from typing import Iterable, Iterator, Tuple

import numpy as np
from scipy.special import i0

from spatialsfs import _random
from spatialsfs.branchingdiffusion import BranchingDiffusion


def sample_positions(
    branching_diffusion: BranchingDiffusion, time: float, seed
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
    seed
        A seed for numpy random generator.

    Notes
    -----
    The positions are modeled as a Brownian bridge, conditional on birth and death
    times and locations. Repeated calls to `positions_at` will generate independent
    draws from the Brownian bridge.

    """
    t_b, t_d, x_b, x_d = branching_diffusion[branching_diffusion.alive_at(time)]
    num_indiv = len(t_b)
    return _random.brownian_bridge(
        time,
        t_b,
        t_d,
        x_b,
        x_d,
        branching_diffusion.diffusion_coefficient,
        _random.raw_distances(num_indiv, branching_diffusion.ndim, seed),
    )


def _generate_sample_positions(
    branching_diffusion: BranchingDiffusion,
    times: Iterable[float],
    init_positions: Iterable[np.ndarray],
    seed: np.random.SeedSequence,
) -> Iterator[np.ndarray]:
    for t, x0 in zip(times, init_positions):
        yield sample_positions(branching_diffusion, t, seed) + x0


def sample(
    branching_diffusion: BranchingDiffusion,
    num_samples: int,
    concentration: float,
    habitat_size: float,
    seed,
    num_centers: int = 1,
    separation: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Randomly generate the contributions to spatially localized samples.

    Parameters
    ----------
    branching_diffusion : BranchingDiffusion
        The branching diffusion to sample from.
    num_samples : int
        The number of (time, position) pairs to sample.
    concentration : float
        The concentration parameter `kappa` of the Von Mises sampling kernel
    habitat_size : float
        The linear size of the habitat.
    seed :
        A seed for numpy random number generation
    num_centers : int
        The number of concentrated sampling centers. Must be 1 or 2.
        (default: 1)
    separation : float
        The distance separating two sampling centers.
        (defaut: 0, ignored if num_centers == 1)

    Returns
    -------
    intensities: np.ndarray
        The number of individuals alive at sampled times weighted by the sampling kernel
        intensities.shape == (num_samples, num_centers)
    importance_weights
        The weights on each sample from the initial position importance sampling.

    """
    if num_centers not in [1, 2]:
        raise ValueError("num_centers must be 1 or 2.")
    seed1, seed2, seed3 = _random.seed_handler(seed).spawn(3)
    times = _random.sample_times(
        num_samples, branching_diffusion.branching_process.final_time, seed1
    )
    init_positions, weights = _random.importance_sample_x0(
        num_samples,
        branching_diffusion.ndim,
        _sampling_scale(branching_diffusion, concentration, separation, habitat_size),
        seed2,
    )
    positions = _generate_sample_positions(
        branching_diffusion, times, init_positions, seed3
    )
    intensities = _intensity(
        positions, num_samples, concentration, habitat_size, num_centers, separation,
    )
    return intensities, weights


def _sampling_scale(
    bd: BranchingDiffusion, concentration: float, separation: float, habitat_size: float
) -> float:
    if concentration == 0.0:
        return habitat_size
    else:
        return min(
            2 * max(bd.scale(), concentration ** (-1 / 2) + separation), habitat_size
        )


def _intensity(
    positions: Iterator[np.ndarray],
    num_samples: int,
    concentration: float,
    habitat_size: float,
    num_centers: int,
    separation: float,
) -> np.ndarray:
    intensities = np.empty((num_samples, num_centers))
    for i, x in enumerate(positions):
        if num_centers == 1:
            intensities[i] = sample_intensity(concentration, habitat_size, x)
        elif num_centers == 2:
            intensities[i] = two_sample_intensity(
                concentration, habitat_size, x, separation
            )
    assert i == num_samples - 1
    return intensities


def sample_intensity(concentration: float, habitat_size: float, x: np.ndarray) -> float:
    """Compute the sample weight according to a Von Mises sampling kernel.

    Parameters
    ----------
    concentration : float
        The concentration parameter kappa of the Von Mises distribution.
        The kernel approaches a Gaussian with sigma^2~kappa for large values of kappa
    habitat_size : float
        The linear size of the habitat.
    x : np.ndarray
        2D array of positions. Shape is (num_indivs, ndim)

    Returns
    -------
    float

    """
    return np.sum(
        np.exp(concentration * np.sum(np.cos(2 * np.pi * x / habitat_size), axis=1))
    ) / (habitat_size * i0(concentration)) ** (x.shape[1])


def two_sample_intensity(
    concentration: float, habitat_size: float, x: np.ndarray, separation: float
) -> Tuple[float, float]:
    """Compute the joint sample intensity for two sampling centers.

    Parameters
    ----------
    concentration : float
        The concentration parameter kappa of the Von Mises distribution.
        The kernel approaches a Gaussian with sigma^2~kappa for large values of kappa
    habitat_size : float
        The linear size of the habitat.
    x : np.ndarray
        2D array of positions. Shape is (num_indivs, ndim)
    separation : float
        The linear separation between the sampling centers.

    Returns
    -------
    Tuple[float, float]
        (intensity_at_center1, intensity_at_center2)

    """
    translation = np.zeros(x.shape[1])
    translation[0] = separation / 2
    return (
        sample_intensity(concentration, habitat_size, x + translation),
        sample_intensity(concentration, habitat_size, x - translation),
    )
