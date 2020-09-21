"""Test spatialsfs.sampling."""
import numpy as np
import pytest

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


@pytest.mark.parametrize("x_0", [-1.0, 0.0, 0.5, np.ones(1)])
@pytest.mark.parametrize("t", [-1.0, 0.0, 0.25, 0.5, 0.6])
def test_sample_weight(small_bd, t, x_0):
    """Test sample_weight."""
    seed = 100
    assert sample_weight(
        small_bd, t, x_0, lambda x: 1.0, seed,
    ) == small_bd.num_alive_at(t)
