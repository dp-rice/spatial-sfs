"""Test spatialsfs.sampling."""

import pytest

from spatialsfs.sampling import positions_at


def test_positions_at(small_bd):
    """Test positions_at."""
    seed = 100
    # Should give empty array when sampled too early or late
    assert len(positions_at(small_bd, -1.0, seed)) == 0
    with pytest.raises(ValueError):
        positions_at(small_bd, small_bd.branching_process.final_time, seed)
    with pytest.raises(ValueError):
        positions_at(small_bd, small_bd.branching_process.final_time + 1.0, seed)
    # Test correct shape
    assert positions_at(small_bd, 0.0, seed).shape == (1, small_bd.ndim)
    assert positions_at(small_bd, 0.25, seed).shape == (1, small_bd.ndim)
    assert positions_at(small_bd, 0.50, seed).shape == (2, small_bd.ndim)
    assert positions_at(small_bd, 0.65, seed).shape == (2, small_bd.ndim)
