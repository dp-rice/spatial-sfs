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
