"""Tests for the branchingdiffusion module."""
from copy import deepcopy

import numpy as np
import pytest

from spatialsfs.branchingdiffusion import BranchingDiffusion
from spatialsfs.branchingprocess import BranchingProcess


@pytest.fixture
def small_bp():
    """Return a simple BranchingProcess with no restarts."""
    parents = np.array([0, 0, 1, 1])
    birth_times = np.array([0.0, 0.0, 0.5, 0.5])
    death_times = np.array([0.0, 0.5, 0.75, np.inf])
    s = 0.05
    return BranchingProcess(parents, birth_times, death_times, s)


@pytest.fixture
def small_bd(small_bp):
    """Return a simple BranchingDiffusion with no restarts."""
    birth_positions = np.array([0.0, 0.0, 0.25, 0.25]).reshape((4, 1))
    death_positions = np.array([0.0, 0.25, 0.35, np.nan]).reshape((4, 1))
    d = 0.5
    return BranchingDiffusion(small_bp, birth_positions, death_positions, d)


# @pytest.fixture
# def large_bd():
#     """Return a more complex BranchingProcess with one restart."""
#     s = 0.05
#     return BranchingProcess(
#         np.array([0, 0, 1, 1, 0, 4, 4]),
#         np.array([0.0, 0.0, 1.0, 1.0, 2.0, 3.5, 3.5]),
#         np.array([0.0, 1.0, 1.5, 2.0, 3.5, 4.0, np.inf]),
#         s,
#     )


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


@pytest.mark.parametrize("time", [-0.5, 0.0, 0.25, 0.5, 0.75])
def test_wrappers(small_bd, small_bp, time):
    """Test alive_at and num_alive_at."""
    assert small_bd.num_alive_at(time) == small_bp.num_alive_at(time)
    np.testing.assert_array_equal(small_bd.alive_at(time), small_bp.alive_at(time))


def test_positions_at():
    """Test positions_at."""
    pass


#     def test_positions_at(self):
#         """Test positions_at."""
#         # Use a round value for mocking gaussians with sd=1
#         mock_rng = mock.Mock()
#         mock_rng.standard_normal.return_value = 1.0

#         # Can't get positions if haven't run position simulations.
#         bd = BranchingDiffusion()
#         with self.assertRaises(RuntimeError):
#             bd.positions_at(1.0, mock_rng)
#         bd.birth_times = self.bd.birth_times.copy()
#         bd.death_times = self.bd.death_times.copy()
#         with self.assertRaises(RuntimeError):
#             bd.positions_at(1.0, mock_rng)

#         # t < 0.0 should give an array with first dimension == zero.
#         self.assertEqual(self.bd.positions_at(-0.5, mock_rng).shape[0], 0)

#         # t == 0.0 should give 0.0
#         np.testing.assert_array_equal(
#             self.bd.positions_at(0.0, mock_rng), np.zeros((1, 1))
#         )

#         # t == 0.25 should give an interpolation
#         t = 0.25
#         expected_position = (
#             (t * 0.3 / 0.5)
#             + np.sqrt(self.bd.diffusion_coefficient * (0.5 - t) * t / 0.5)
#         ).reshape((-1, 1))
#         np.testing.assert_array_equal(
#             self.bd.positions_at(t, mock_rng), expected_position
#         )

#         # t == 0.6 should be length 2
#         t = 0.6
#         ep1 = (
#             0.3
#             + ((t - 0.5) * (-0.4) / 0.5)
#             + np.sqrt(self.bd.diffusion_coefficient * (1.0 - t) * (t - 0.5) / 0.5)
#         )
#         ep2 = (
#             0.3
#             + ((t - 0.5) * 1.0 / 1.0)
#             + np.sqrt(self.bd.diffusion_coefficient * (1.5 - t) * (t - 0.5) / 1.0)
#         )
#         expected_position = np.vstack([ep1, ep2]).reshape((-1, 1))
#         np.testing.assert_array_equal(
#             self.bd.positions_at(t, mock_rng), expected_position
#         )
