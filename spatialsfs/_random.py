import numpy as np


def raw_times(num_steps: int, seedseq: np.random.SeedSequence) -> np.ndarray:
    """Generate random (unscaled) exponentially distributed intervals.

    Parameters
    ----------
    num_steps : int
        The number of times to generate
    seedseq : np.random.SeedSequence
        A random seed

    Returns
    -------
    np.ndarray

    """
    return np.random.default_rng(seedseq).standard_exponential(size=num_steps)


def num_offspring(
    num_steps: int, s: float, seedseq: np.random.SeedSequence
) -> np.ndarray:
    """Generate random number of offspring.

    Parameters
    ----------
    num_steps : int
        The number of sets of offspring to generate.
    s : float
        The selection coefficient
    seedseq : np.random.SeedSequence
        A random seed

    Returns
    -------
    np.ndarray

    """
    return 2 * np.random.default_rng(seedseq).binomial(1, (1 - s) / 2, size=num_steps)


def parent_choices(num_steps: int, seedseq: np.random.SeedSequence) -> np.ndarray:
    """Generate random uniform floats in [0, 1] representing parent choices.

    Parameters
    ----------
    num_steps : int
        The number of times to generate
    seedseq : np.random.SeedSequence
        A random seed

    Returns
    -------
    np.ndarray

    """
    return np.random.default_rng(seedseq).random(size=num_steps)


def raw_distances(
    num_indivs: int, ndim: int, seedseq: np.random.SeedSequence
) -> np.ndarray:
    """Generate random ndim-dimensional multivariate Gaussian distances.

    Parameters
    ----------
    num_indivs : int
        The number of individuals to simulate.
    ndim : int
        The number of spatial dimensions
    seedseq : np.random.SeedSequence
        A random seed

    Returns
    -------
    np.ndarray
        Shape is (num_indivs, ndim)

    """
    rng = np.random.default_rng(seedseq)
    return rng.standard_normal(size=(num_indivs, ndim))
