"""Test spatial-sfs.simulations."""


from unittest import TestCase, mock

import numpy as np
import pytest

import spatialsfs.simulations as sims
from spatialsfs.simulations import _generate_tree, branch


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


# New parameterized test (parameterized on seed).
# assert root exists in all output
# lengths ok?
# births in order
# assertTrue(0 <= parents[i] and parents[i] < i)
# assertEqual(birth_times[i], death_times[parents[i]])


class TestSimulations(TestCase):
    """TestSimulations."""

    def test_simulate_positions(self):
        """Test simulate_positions."""
        n = 3
        scale = 2.0
        d = scale ** 2
        parents = [None, 0, 0]
        lifespans = np.array([1.0, 2.0, 3.0])
        # Simulations that reached max-steps
        lifespans_nan = np.array([1.0, np.nan, np.nan])
        mock_rng = mock.Mock()

        # Should complain about invalid ndims
        for ndims in [-1, 0]:
            with self.assertRaises(ValueError):
                bp, dp = sims.simulate_positions(d, ndims, parents, lifespans, mock_rng)

        for ndims in range(1, 3):
            mock_rng.standard_normal.return_value = np.ones((n, ndims))
            bp_expect = scale * np.vstack([[0.0, 1.0, 1.0]] * ndims).T
            dp_expect = bp_expect + scale * np.sqrt(lifespans)[:, None]
            bp, dp = sims.simulate_positions(d, ndims, parents, lifespans, mock_rng)
            np.testing.assert_array_equal(bp, bp_expect)
            np.testing.assert_array_equal(dp, dp_expect)

            # Simulations that reached max-steps
            dp_nan = bp_expect + scale * np.sqrt(lifespans_nan)[:, None]
            bp, dp = sims.simulate_positions(d, ndims, parents, lifespans_nan, mock_rng)
            np.testing.assert_array_equal(bp, bp_expect)
            np.testing.assert_array_equal(dp, dp_nan)
