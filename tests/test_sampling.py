"""Test spatialsfs.sampling."""
import numpy as np
import pytest
from scipy.stats import vonmises

from spatialsfs.sampling import sample_positions, sample_weight


def test_sample_positions(small_bd):
    """Test sample_positions."""
    seed = 100
    # Should give empty array when sampled too early or late
    assert len(sample_positions(small_bd, -1.0, seed)) == 0
    with pytest.raises(ValueError):
        sample_positions(small_bd, small_bd.branching_process.final_time, seed)
    with pytest.raises(ValueError):
        sample_positions(small_bd, small_bd.branching_process.final_time + 1.0, seed)
    # Test correct shape
    assert sample_positions(small_bd, 0.0, seed).shape == (1, small_bd.ndim)
    assert sample_positions(small_bd, 0.25, seed).shape == (1, small_bd.ndim)
    assert sample_positions(small_bd, 0.50, seed).shape == (2, small_bd.ndim)
    assert sample_positions(small_bd, 0.65, seed).shape == (2, small_bd.ndim)


@pytest.mark.parametrize("kappa", [0.01, 1, 100.0])
@pytest.mark.parametrize("ndim", [1, 2])
def test_sample_weight(ndim, kappa):
    """Test sample_weight."""
    x = np.arange(10.0).reshape(-1, ndim)
    L = 5.0
    assert np.isclose(
        sample_weight(kappa, L, x),
        np.sum(np.product(vonmises(kappa, scale=L / (4 * np.pi)).pdf(x), axis=1)),
    )
