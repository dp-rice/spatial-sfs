"""Tests for the branchingdiffusion module."""
from copy import deepcopy

import numpy as np
import pytest

from spatialsfs.branchingdiffusion import BranchingDiffusion


def test_setting_attributes(small_bp):
    """Test __init__."""
    birth_positions = np.array([0.0, 0.0, 0.25, 0.25]).reshape((4, 1))
    death_positions = np.array([0.0, 0.25, 0.35, np.nan]).reshape((4, 1))
    d = 0.5
    bd = BranchingDiffusion(deepcopy(small_bp), birth_positions, death_positions, d)
    assert bd.branching_process == small_bp
    assert np.array_equal(bd.birth_positions, birth_positions, equal_nan=True)
    assert np.array_equal(bd.death_positions, death_positions, equal_nan=True)
    assert bd.diffusion_coefficient == d
    assert bd.ndim == 1


@pytest.mark.parametrize("ndim", [1, 2])
def test_type_checking(small_bp, ndim):
    """Test type checking in __init__."""
    n = len(small_bp)
    d = 0.5
    with pytest.raises(TypeError):
        BranchingDiffusion(
            small_bp,
            np.zeros((n, ndim), dtype=int),
            np.zeros((n, ndim), dtype=float),
            d,
        )
    with pytest.raises(TypeError):
        BranchingDiffusion(
            small_bp,
            np.zeros((n, ndim), dtype=float),
            np.zeros((n, ndim), dtype=int),
            d,
        )


@pytest.mark.parametrize("ndim", [1, 2])
def test_length_checking(small_bp, ndim):
    """The input arrays must all be the same length as branching_process."""
    n = len(small_bp)
    d = 0.5
    with pytest.raises(ValueError):
        BranchingDiffusion(small_bp, np.zeros((n - 1, ndim)), np.zeros((n, ndim)), d)
    with pytest.raises(ValueError):
        BranchingDiffusion(small_bp, np.zeros((n, ndim)), np.zeros((n - 1, ndim)), d)


def test_eq(small_bd):
    """Test that equality means equality of attributes."""
    bd = deepcopy(small_bd)
    assert bd == small_bd
    bd.diffusion_coefficient /= 2
    assert bd != small_bd
    bd = deepcopy(small_bd)
    bd.birth_positions[-1] /= 2
    assert bd != small_bd
    bd = deepcopy(small_bd)
    bd.death_positions[1] /= 2
    assert bd != small_bd
    bd = deepcopy(small_bd)
    bd.branching_process.birth_times[-1] /= 2
    assert bd != small_bd


def test_len(small_bd, small_bp):
    """Test __len__."""
    assert len(small_bd) == len(small_bp)


def test_num_restarts(small_bd, small_bp):
    """Test num_restarts."""
    assert small_bd.num_restarts() == small_bp.num_restarts()


@pytest.mark.parametrize("time", [-0.5, 0.0, 0.25, 0.5, 0.74])
def test_wrappers(small_bd, small_bp, time):
    """Test alive_at and num_alive_at."""
    assert small_bd.num_alive_at(time) == small_bp.num_alive_at(time)
    np.testing.assert_array_equal(small_bd.alive_at(time), small_bp.alive_at(time))
