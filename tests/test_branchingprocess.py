"""Test the branchingprocess module."""
from copy import deepcopy

import numpy as np
import pytest

from spatialsfs.branchingprocess import BranchingProcess, branch, separate_restarts


@pytest.fixture
def small_bp():
    """Return a simple BranchingProcess with no restarts."""
    parents = np.array([0, 0, 1, 1])
    birth_times = np.array([0.0, 0.0, 0.5, 0.5])
    death_times = np.array([0.0, 0.5, 0.75, np.nan])
    s = 0.05
    return BranchingProcess(parents, birth_times, death_times, s)


@pytest.fixture
def large_bp():
    """Return a more complex BranchingProcess with one restart."""
    s = 0.05
    return BranchingProcess(
        np.array([0, 0, 1, 1, 0, 4, 4]),
        np.array([0.0, 0.0, 1.0, 1.0, 2.0, 3.5, 3.5]),
        np.array([0.0, 1.0, 1.5, 2.0, 3.5, 4.0, np.nan]),
        s,
    )


def test_setting_attributes():
    """Test __init__."""
    parents = np.array([0, 0, 1, 1])
    birth_times = np.array([0.0, 0.0, 0.5, 0.5])
    death_times = np.array([0.0, 0.5, 0.75, np.nan])
    s = 0.05
    bp = BranchingProcess(parents, birth_times, death_times, s)
    assert np.array_equal(bp.parents, parents, equal_nan=True)
    assert np.array_equal(bp.birth_times, birth_times, equal_nan=True)
    assert np.array_equal(bp.death_times, death_times, equal_nan=True)
    assert bp.selection_coefficient == s


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
        np.array([0.0, 1.5, 2.0, np.nan]),
        large_bp.selection_coefficient,
    )
    with pytest.raises(StopIteration):
        next(bp_iterator)


def test_branch_checks_s():
    """Test that branch does not allow s<=0 or s>=1."""
    with pytest.raises(ValueError):
        branch(10, -0.05, 100)
    with pytest.raises(ValueError):
        branch(10, 0.0, 100)
    with pytest.raises(ValueError):
        branch(10, 1.0, 100)
    with pytest.raises(ValueError):
        branch(10, 1.1, 100)