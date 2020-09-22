"""Test _random.py module."""
import numpy as np
import pytest
from scipy.stats import multivariate_normal

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


def test_sample_times():
    """Test sample_time."""
    num_samples = 100
    max_time = 1e4
    seed = 1
    samples = _random.sample_times(num_samples, max_time, seed)
    print(samples)
    assert len(samples) == num_samples
    assert max(samples) < max_time
    assert max(samples) > 1.0
    assert min(samples) > 0.0


@pytest.mark.parametrize("ndim", [1, 2])
def test_gaussian_weight(ndim):
    """Check that Gaussian weights are proportional to one over normal pdf."""
    x = np.arange(10.0).reshape((-1, ndim))
    scale = 2.0
    ratio = _random._gaussian_weight(x, scale) * multivariate_normal(
        mean=np.zeros(ndim), cov=np.eye(ndim) * scale ** 2
    ).pdf(x)
    assert np.allclose(ratio, ratio[0])


@pytest.mark.parametrize("ndim", [1, 2])
def test_importance_sample_x0(ndim):
    """Test return shape of importance_sample_x0."""
    num_samples = 10
    scale = 0.5
    x_0, weights = _random.importance_sample_x0(num_samples, ndim, scale, 1)
    assert x_0.shape == (num_samples, ndim)
    assert weights.shape == (num_samples,)
