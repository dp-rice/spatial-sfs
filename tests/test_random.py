"""Test _random.py module."""
import numpy as np
import pytest

from spatialsfs import _random


@pytest.mark.parametrize("seed", [1, 3931048, np.random.SeedSequence(50)])
def test_seed_handler(seed):
    """Test seed_handler."""
    seedseq = _random.seed_handler(seed)
    if isinstance(seed, np.random.SeedSequence):
        assert seedseq is seed
    else:
        assert seedseq.entropy == seed


@pytest.mark.parametrize("t0", [0.0, 1.0])
@pytest.mark.parametrize("x0", [0.0, 1.0])
@pytest.mark.parametrize("d", [0.5, 1.0])
@pytest.mark.parametrize("t", [0.0, 0.25, 1.0])
@pytest.mark.parametrize("ndim", [1, 2])
@pytest.mark.parametrize("num_indiv", [0, 1, 10])
def test_brownian_bridge(num_indiv, ndim, t, d, x0, t0):
    """Test brownian_bridge."""
    t_a = np.zeros(num_indiv) + t0
    t_b = np.ones(num_indiv) + t0
    x_a = np.zeros((num_indiv, ndim)) + x0
    x_b = np.ones((num_indiv, ndim)) + x0
    raw_distances = np.ones((num_indiv, ndim))
    expected = (t + np.sqrt(t * (1 - t) * d)) * np.ones((num_indiv, ndim)) + x0
    bb = _random.brownian_bridge(t + t0, t_a, t_b, x_a, x_b, d, raw_distances)
    np.testing.assert_array_equal(bb, expected)
