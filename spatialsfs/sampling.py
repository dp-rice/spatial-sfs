"""Functions for sampling positions and the SFS."""

import numpy as np

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


def sample_weight(
    bd: BranchingDiffusion, t: float, x_0, sampling_kernel, seed
) -> float:
    """Compute the weight a BranchingDiffusion contributes to a sample.

    Parameters
    ----------
    bd : BranchingDiffusion
        The branching diffusion to sample from.
    t : float
        The time at which to take the sample.
    x_0 : np.arraylike
        The initial position of the branching diffusion
    sampling_kernel :
        The kernel used to collect the sample.
        Must be a function with signature:
        sampling_kernel(x: np.ndarray) -> float
    seed :
        A seed for numpy random number generation

    Returns
    -------
    float
        The weight contributed to the sample.

    """
    return sum(sampling_kernel(x_0 + x) for x in sample_positions(bd, t, seed))
