"""Test spatialsfs.simulations."""

import numpy as np
import pytest

from spatialsfs.simulations import (
    _generate_positions,
    _generate_tree,
    branch,
    diffuse,
    simulate_branching_diffusion,
)


@pytest.fixture
def seedseq():
    """Return a seed sequence."""
    return np.random.SeedSequence(100)


def test_branch_checks_s(seedseq):
    """Test that branch does not allow s<=0 or s>=1."""
    with pytest.raises(ValueError):
        branch(10, -0.05, seedseq)
    with pytest.raises(ValueError):
        branch(10, 0.0, seedseq)
    with pytest.raises(ValueError):
        branch(10, 1.0, seedseq)
    with pytest.raises(ValueError):
        branch(10, 1.1, seedseq)


def test_diffuse_checks_ndim(small_bp, seedseq):
    """Test that diffuse does not allow ndim<=0."""
    with pytest.raises(ValueError):
        diffuse(small_bp, 0, 0.5, seedseq)
    with pytest.raises(ValueError):
        diffuse(small_bp, -1, 0.5, seedseq)


def test_diffuse_checks_d(small_bp, seedseq):
    """Test that diffuse does not allow d<=0."""
    with pytest.raises(ValueError):
        diffuse(small_bp, 1, -0.05, seedseq)
    with pytest.raises(ValueError):
        diffuse(small_bp, 2, 0.0, seedseq)


def test_generate_tree_asserts():
    """Test that generate_tree checks array lengths and types."""
    with pytest.raises(AssertionError):
        _generate_tree(np.zeros(4), np.zeros(3, dtype=int), np.zeros(3))
    with pytest.raises(AssertionError):
        _generate_tree(np.zeros(4), np.zeros(4, dtype=int), np.zeros(3))
    with pytest.raises(AssertionError):
        _generate_tree(np.zeros(4), np.zeros(4, dtype=float), np.zeros(4))


def test_generate_positions_asserts(small_bp):
    """Test that generate_tree checks array shapes and types."""
    length = len(small_bp)
    d = 0.5
    # Too long
    with pytest.raises(AssertionError):
        _generate_positions(small_bp, np.zeros((length + 1, 1)), d)
    # Wrong shape
    with pytest.raises(AssertionError):
        _generate_positions(small_bp, np.zeros(length), d)
    # Wrong type
    with pytest.raises(AssertionError):
        _generate_positions(small_bp, np.zeros((length, 1), dtype=int), d)


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
                np.array([0.0, 1.0, 1.5, 1.5, 1.5, 1.5]),
            ),
        ),
    ],
)
def test_generate_tree(raw_times, num_offspring, parent_choices, expected):
    """Test _generate_tree on a variety of inputs."""
    output = _generate_tree(raw_times, num_offspring, parent_choices)
    np.testing.assert_equal(output, expected)


@pytest.mark.parametrize("ndim", [1, 2])
def test_generate_positions(small_bp, ndim):
    """Test _generate_positions with ones for rng output."""
    simple_distances = np.ones((len(small_bp), ndim))
    d = 4.0
    bp_expected = np.array([[0.0, 0.0, np.sqrt(0.5 * d), np.sqrt(0.5 * d)]] * ndim).T
    dp_expected = np.array(
        [
            [
                0.0,
                np.sqrt(0.5 * d),
                np.sqrt(0.5 * d) + np.sqrt(0.25 * d),
                np.sqrt(0.5 * d) + np.sqrt(0.25 * d),
            ]
        ]
        * ndim
    ).T
    bp, dp = _generate_positions(small_bp, simple_distances, d)
    np.testing.assert_array_equal(bp, bp_expected)
    np.testing.assert_array_equal(dp, dp_expected)


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("ndim", [1, 2])
def test_simulate_branching_diffusion(ndim, seed):
    """Test that everything runs."""
    num_steps = 100
    s = 0.05
    d = 2.0
    bd = simulate_branching_diffusion(num_steps, s, ndim, d, seed)
    bp = bd.branching_process
    assert len(bd) == len(bp)
    num_indivs = len(bd)
    assert bd.birth_positions.shape == (num_indivs, ndim)
    assert bd.death_positions.shape == (num_indivs, ndim)
    assert bp.parents.shape == (num_indivs,)
    assert bp.birth_times.shape == (num_indivs,)
    assert bp.death_times.shape == (num_indivs,)
    assert bd.diffusion_coefficient == d
    assert bp.selection_coefficient == s
    for i in range(1, num_indivs):
        assert 0 <= bp.parents[i] < i
        assert bp.birth_times[i - 1] <= bp.birth_times[i]
        assert bp.parents[i] == 0 or bp.birth_times[i] == bp.death_times[bp.parents[i]]
        assert np.all(bd.birth_positions[i] == bd.death_positions[bp.parents[i]])
