"""Compute theoretical quantities."""
from typing import Callable, Tuple

import numpy as np
from numpy.random import SeedSequence
from numpy.random import default_rng as rng


def sample_gaussian(num_samples: int, ndim: int, seed) -> np.ndarray:
    """Generate random ndim-dimensional multivariate Gaussian locations.

    Parameters
    ----------
    num_samples : int
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
    return rng(seed).standard_normal(size=(num_samples, ndim))


def gaussian_integral(
    f: Callable, num_samples: int, num_vars: int, ndim: int, seed, **kwargs
) -> Tuple[float, float]:
    """Monte Carlo integrate a function against a multidimensional gaussian density.

    Parameters
    ----------
    f : Callable
        The function to integrate.
        Takes num_vars (num_samples, ndim) arrays and return an (num_samples) array
    num_samples : int
        The number of independent samples to take
    num_vars : int
        The number of multidimensional variables f takes
    ndim : int
        The number of dimensions of each variable
    seed :
        Valid seed for numpy random
    kwargs :
        Keyword arguments to pass to f

    Returns
    -------
    Tuple[float, float]
        sample_mean, standard_error

    """
    locations = [
        sample_gaussian(num_samples, ndim, child_seed)
        for child_seed in SeedSequence(seed).spawn(num_vars)
    ]
    samples = f(*locations, **kwargs)
    return np.mean(samples), np.std(samples) / np.sqrt(num_samples)


def a2(x: np.ndarray, d: float = 1.0) -> np.ndarray:
    """Compute the second taylor coefficient."""
    norm_x = np.sum(x ** 2, axis=1)
    return 2 * np.pi * np.exp(-norm_x / 2) / (1 + d * norm_x)


def a3(x1: np.ndarray, x2: np.ndarray, d: float = 1.0) -> np.ndarray:
    """Compute the third taylor coefficient."""
    norm_x1 = np.sum(x1 ** 2, axis=1)
    norm_x2 = np.sum(x2 ** 2, axis=1)
    norm_x12 = np.sum((x1 + x2) ** 2, axis=1)
    return (
        (2 * np.pi) ** 2
        * np.exp(-norm_x12 / 2)
        / ((2 + 2 * d * norm_x12) * (3 + d * (norm_x12 + norm_x1 + norm_x2)))
    )
