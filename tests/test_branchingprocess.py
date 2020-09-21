"""Test the branchingprocess module."""
from copy import deepcopy

import numpy as np
import pytest

from spatialsfs.branchingprocess import BranchingProcess, separate_restarts


def test_setting_attributes():
    """Test __init__."""
    parents = np.array([0, 0, 1, 1])
    birth_times = np.array([0.0, 0.0, 0.5, 0.5])
    death_times = np.array([0.0, 0.5, 0.75, 0.75])
    s = 0.05
    bp = BranchingProcess(parents, birth_times, death_times, s)
    assert np.array_equal(bp.parents, parents, equal_nan=True)
    assert np.array_equal(bp.birth_times, birth_times, equal_nan=True)
    assert np.array_equal(bp.death_times, death_times, equal_nan=True)
    assert bp.selection_coefficient == s
    assert bp.final_time == np.max(death_times[np.isfinite(death_times)])


def test_type_checking():
    """Test type checking in __init__."""
    parents = np.array([0, 0, 1, 1])
    birth_times = np.array([0.0, 0.0, 0.5, 0.5])
    death_times = np.array([0.0, 0.5, 0.75, 1.0])
    s = 0.05
    with pytest.raises(TypeError):
        BranchingProcess(np.zeros(4, dtype=float), birth_times, death_times, s)
    with pytest.raises(TypeError):
        BranchingProcess(parents, np.zeros(4, dtype=int), death_times, s)
    with pytest.raises(TypeError):
        BranchingProcess(parents, birth_times, np.zeros(4, dtype=int), s)


def test_length_checking():
    """The input arrays must all bhe the same length."""
    parents = np.array([0, 0, 1, 1])
    s = 0.05
    birth_times = np.array([0.0, 0.0, 0.5])
    death_times = np.array([0.0, 0.5, 0.75, 1.0])
    with pytest.raises(ValueError):
        BranchingProcess(parents, birth_times, death_times, s)
    birth_times = np.array([0.0, 0.0, 0.5, 0.5])
    death_times = np.array([0.0, 0.5, 0.75])
    with pytest.raises(ValueError):
        BranchingProcess(parents, birth_times, death_times, s)


def test_root_checking():
    """The first entry in each attribute array must be zero: the root of the tree."""
    s = 0.05
    with pytest.raises(ValueError):
        BranchingProcess(np.array([1]), np.array([0.0]), np.array([0.0]), s)
    with pytest.raises(ValueError):
        BranchingProcess(np.array([0]), np.array([1.0]), np.array([0.0]), s)
    with pytest.raises(ValueError):
        BranchingProcess(np.array([0]), np.array([0.0]), np.array([1.0]), s)


def test_eq(small_bp):
    """Equality means equality of attributes."""
    bp2 = deepcopy(small_bp)
    assert small_bp == bp2
    bp2.parents[-1] = 0
    assert small_bp != bp2
    bp2 = deepcopy(small_bp)
    bp2.birth_times[2] *= 2
    assert small_bp != bp2
    bp2 = deepcopy(small_bp)
    bp2.death_times[-2] *= 2
    assert small_bp != bp2
    bp2 = deepcopy(small_bp)
    bp2.death_times[-2] *= 2
    assert small_bp != bp2
    bp2 = deepcopy(small_bp)
    bp2.selection_coefficient *= 2
    assert small_bp != bp2


def test_getitem(small_bp):
    """Getting an index means slicing the arrays."""
    assert small_bp[0] == (np.array([0]), np.array([0.0]), np.array([0.0]))
    assert small_bp[1:2] == (
        small_bp.parents[1:2],
        small_bp.birth_times[1:2],
        small_bp.death_times[1:2],
    )


def test_num_restarts(small_bp, large_bp):
    """Test the number of restarts."""
    assert small_bp.num_restarts() == 1
    assert large_bp.num_restarts() == 2


def test_len(small_bp, large_bp):
    """Test __len__ function."""
    assert small_bp.parents.shape == (len(small_bp),)
    assert large_bp.parents.shape == (len(large_bp),)


@pytest.mark.parametrize(
    "time,expected",
    [
        (-0.5, np.array([False, False, False, False])),
        (0.0, np.array([False, True, False, False])),
        (0.25, np.array([False, True, False, False])),
        (0.50, np.array([False, False, True, True])),
        (0.60, np.array([False, False, True, True])),
    ],
)
def test_alive_at(small_bp, time, expected):
    """Test alive_at."""
    np.testing.assert_array_equal(small_bp.alive_at(time), expected)


def test_alive_at_too_late(small_bp, large_bp):
    """Test that exception is raised for a time that's at or beyond the final time."""
    with pytest.raises(ValueError):
        small_bp.alive_at(small_bp.final_time)
    with pytest.raises(ValueError):
        small_bp.alive_at(small_bp.final_time + 1.0)
    with pytest.raises(ValueError):
        large_bp.alive_at(large_bp.final_time)
    with pytest.raises(ValueError):
        large_bp.alive_at(large_bp.final_time + 1.0)


@pytest.mark.parametrize(
    "time,expected",
    [(-0.5, 0), (0.0, 1), (0.5, 1), (1.0, 2), (1.25, 2), (1.5, 1), (2.0, 1), (3.75, 2)],
)
def test_num_alive_at(large_bp, time, expected):
    """Test num_alive_at."""
    assert large_bp.num_alive_at(time) == expected


def test_num_alive_at_too_late(small_bp, large_bp):
    """Test that exception is raised when given a time that's beyond the final time."""
    with pytest.raises(ValueError):
        small_bp.num_alive_at(0.76)
    with pytest.raises(ValueError):
        large_bp.num_alive_at(4.1)


def test_separate_restarts(large_bp):
    """Test separating restarts."""
    bp_iterator = separate_restarts(large_bp)
    assert next(bp_iterator) == BranchingProcess(
        np.array([0, 0, 1, 1]),
        np.array([0.0, 0.0, 1.0, 1.0]),
        np.array([0.0, 1.0, 1.5, 2.0]),
        large_bp.selection_coefficient,
    )
    assert next(bp_iterator) == BranchingProcess(
        np.array([0, 0, 1, 1]),
        np.array([0.0, 0.0, 1.5, 1.5]),
        np.array([0.0, 1.5, 2.0, 2.0]),
        large_bp.selection_coefficient,
    )
    with pytest.raises(StopIteration):
        next(bp_iterator)
