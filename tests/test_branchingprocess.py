"""Test the branchingprocess module."""
from copy import deepcopy

import numpy as np
import pytest

from spatialsfs.branchingprocess import (
    BranchingProcess,
    _generate_tree,
    branch,
    separate_restarts,
)


@pytest.fixture
def small_bp():
    """Return a simple BranchingProcess with no restarts."""
    parents = np.array([0, 0, 1, 1])
    birth_times = np.array([0.0, 0.0, 0.5, 0.5])
    death_times = np.array([0.0, 0.5, 0.75, np.inf])
    s = 0.05
    return BranchingProcess(parents, birth_times, death_times, s)


@pytest.fixture
def large_bp():
    """Return a more complex BranchingProcess with one restart."""
    s = 0.05
    return BranchingProcess(
        np.array([0, 0, 1, 1, 0, 4, 4]),
        np.array([0.0, 0.0, 1.0, 1.0, 2.0, 3.5, 3.5]),
        np.array([0.0, 1.0, 1.5, 2.0, 3.5, 4.0, np.inf]),
        s,
    )


def test_setting_attributes():
    """Test __init__."""
    parents = np.array([0, 0, 1, 1])
    birth_times = np.array([0.0, 0.0, 0.5, 0.5])
    death_times = np.array([0.0, 0.5, 0.75, np.inf])
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
    """Test that exception is raised when given a time that's beyond the final time."""
    with pytest.raises(ValueError):
        small_bp.alive_at(0.76)
    with pytest.raises(ValueError):
        large_bp.alive_at(4.1)


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
        np.array([0.0, 1.5, 2.0, np.inf]),
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


def test_generate_tree_asserts():
    """Test that generate_tree checks array lengths and types."""
    with pytest.raises(AssertionError):
        _generate_tree(np.zeros(4), np.zeros(3, dtype=int), np.zeros(3))
    with pytest.raises(AssertionError):
        _generate_tree(np.zeros(4), np.zeros(4, dtype=int), np.zeros(3))
    with pytest.raises(AssertionError):
        _generate_tree(np.zeros(4), np.zeros(4, dtype=float), np.zeros(4))


@pytest.mark.parametrize(
    "raw_times,num_offspring,parent_choices,expected",
    [
        (
            np.ones(4),
            np.zeros(4, dtype=int),
            0.5 * np.ones(4),
            (
                np.zeros(4 + 1, dtype=int),
                np.array([0.0, 0.0, 1.0, 2.0, 3.0]),
                np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
            ),
        ),
        (
            np.ones(4),
            np.array([2, 0, 0, 0]),
            0.25 * np.ones(4),
            (
                np.array([0, 0, 1, 1, 0]),
                np.array([0.0, 0.0, 1.0, 1.0, 2.5]),
                np.array([0.0, 1.0, 1.5, 2.5, 3.5]),
            ),
        ),
        (
            np.ones(4),
            np.array([2, 0, 0, 0]),
            np.array([0.25, 0.75, 0.25, 0.25]),
            (
                np.array([0, 0, 1, 1, 0]),
                np.array([0.0, 0.0, 1.0, 1.0, 2.5]),
                np.array([0.0, 1.0, 2.5, 1.5, 3.5]),
            ),
        ),
        (
            np.ones(4) / 2,
            np.array([2, 0, 0, 0]),
            np.array([0.25, 0.75, 0.25, 0.25]),
            (
                np.array([0, 0, 1, 1, 0]),
                np.array([0.0, 0.0, 1.0, 1.0, 2.5]) / 2,
                np.array([0.0, 1.0, 2.5, 1.5, 3.5]) / 2,
            ),
        ),
        (
            np.ones(2),
            np.array([2, 2]),
            np.array([0.25, 0.8]),
            (
                np.array([0, 0, 1, 1, 3, 3]),
                np.array([0.0, 0.0, 1.0, 1.0, 1.5, 1.5]),
                np.array([0.0, 1.0, np.inf, 1.5, np.inf, np.inf]),
            ),
        ),
    ],
)
def test_generate_tree(raw_times, num_offspring, parent_choices, expected):
    """Test _generate_tree on a variety of inputs."""
    output = _generate_tree(raw_times, num_offspring, parent_choices)
    np.testing.assert_equal(output, expected)
