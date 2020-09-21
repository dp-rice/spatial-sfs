import numpy as np


def seed_handler(seed) -> np.random.SeedSequence:
    """Return a SeedSequence given a SeedSequence or valid source of entropy."""
    if isinstance(seed, np.random.SeedSequence):
        return seed
    else:
        return np.random.SeedSequence(seed)


def raw_times(num_steps: int, seed) -> np.ndarray:
    """Generate random (unscaled) exponentially distributed intervals.

    Parameters
    ----------
    num_steps : int
        The number of times to generate
    seedseq
        A random seed

    Returns
    -------
    np.ndarray

    """
    return np.random.default_rng(seed).standard_exponential(size=num_steps)


def num_offspring(num_steps: int, s: float, seed) -> np.ndarray:
    """Generate random number of offspring.

    Parameters
    ----------
    num_steps : int
        The number of sets of offspring to generate.
    s : float
        The selection coefficient
    seed
        A random seed

    Returns
    -------
    np.ndarray

    """
    return 2 * np.random.default_rng(seed).binomial(1, (1 - s) / 2, size=num_steps)


def parent_choices(num_steps: int, seed):
    """Generate random uniform floats in [0, 1] representing parent choices.

    Parameters
    ----------
    num_steps : int
        The number of times to generate
    seed
        A random seed

    Returns
    -------
    np.ndarray

    """
    return np.random.default_rng(seed).random(size=num_steps)


def raw_distances(num_indivs: int, ndim: int, seed) -> np.ndarray:
    """Generate random ndim-dimensional multivariate Gaussian distances.

    Parameters
    ----------
    num_indivs : int
        The number of individuals to simulate.
    ndim : int
        The number of spatial dimensions
    seed
        A random seed

    Returns
    -------
    np.ndarray
        Shape is (num_indivs, ndim)

    """
    rng = np.random.default_rng(seed)
    return rng.standard_normal(size=(num_indivs, ndim))


def brownian_bridge(
    t: float,
    t_a: np.ndarray,
    t_b: np.ndarray,
    x_a: np.ndarray,
    x_b: np.ndarray,
    diffusion_coefficient: float,
    raw_distances: np.ndarray,
) -> np.ndarray:
    """Return random positions drawn from n independent Brownian bridges.

    Parameters
    ----------
    t : float
        The time at which to sample the Brownian bridges.
    t_a : np.ndarray
        1D array with the initial times of the bridges.
        Shape is (n,)
    t_b : np.ndarray
        1D array with the final times of the bridges.
        Shape is (n,)
    x_a : np.ndarray
        2D array with the initial positions of the bridges.
        Shape is (n, ndim)
    x_b : np.ndarray
        2D array with the initial positions of the bridges.
        Shape is (n, ndim)
    diffusion_coefficient : float
        The diffusion coefficient of the brownian bridge.
    raw_distances : np.ndarray
        Standard normal random raw distances.

    Returns
    -------
    np.ndarray
        The positions at t. Shape is (n, ndim).
    """
    assert x_a.shape == raw_distances.shape
    means = x_a + (x_b - x_a) * ((t - t_a) / (t_b - t_a))[:, None]
    variances = diffusion_coefficient * (t_b - t) * (t - t_a) / (t_b - t_a)
    return means + np.sqrt(variances)[:, None] * raw_distances
