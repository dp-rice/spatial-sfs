"""Functions for sampling positions and the SFS."""

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


def sample_weight(concentration: float, habitat_size: float, x: np.ndarray) -> float:
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
        The sample weight

    """
    return np.sum(
        np.exp(concentration * np.sum(np.cos(4 * np.pi * x / habitat_size), axis=1))
    ) / (habitat_size * i0(concentration) / 2) ** (x.shape[1])
