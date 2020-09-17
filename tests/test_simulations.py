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


# class TestBranchingDiffusion(TestCase):
#     """Test initializing, loading, saving, and equality of BrachingDiffusion class."""

#     def setUp(self):
#         """Set up a small instance of BranchingDiffusion."""
#         self.bd = BranchingDiffusion()
#         self.bd.parents = [None, 0, 0]
#         self.bd.birth_times = np.array([0.0, 0.5, 0.5])
#         self.bd.death_times = np.array([0.5, 1.0, 1.5])
#         # 1D positions
#         self.bd.ndim = 1
#         self.bd.birth_positions = np.array([0.0, 0.3, 0.3]).reshape(3, self.bd.ndim)
#         self.bd.death_positions = np.array([0.3, -0.1, 1.3]).reshape(3, self.bd.ndim)
#         self.bd.selection_coefficient = 0.05
#         self.bd.diffusion_coefficient = 0.75
#         self.bd.num_total = 3
#         self.bd.num_max = 2
#         self.bd.extinction_time = 1.5

#     @mock.patch(
#         "spatialsfs.branchingdiffusion.simulations.simulate_positions", autospec=True
#     )
#     def test_simulate_positions(self, mock_sim):
#         """Test simulate_positions method (mocking simulation code)."""
#         mock_sim.return_value = (
#             self.bd.birth_positions,
#             self.bd.death_positions,
#         )
#         rng = np.random.default_rng()
#         # Set up bd as though we'd run simulate_times
#         bd = deepcopy(self.bd)
#         bd.diffusion_coefficient = None
#         bd.birth_positions = np.array([], dtype=float)
#         bd.death_positions = np.array([], dtype=float)
#         # Run simulations
#         d = self.bd.diffusion_coefficient
#         ndim = self.bd.ndim
#         bd.simulate_positions(d, ndim, rng)
#         # Assert called simulate_positions with correct params
#         mock_sim.assert_called_once()
#         self.assertEqual(mock_sim.call_args[0][:3], (d, ndim, self.bd.parents))
#         intervals = self.bd.death_times - self.bd.birth_times
#         np.testing.assert_array_equal(intervals, mock_sim.call_args[0][3])
#         self.assertEqual(bd.diffusion_coefficient, d)
#         self.assertEqual(bd.ndim, ndim)
#         # Should assign simulation output
#         np.testing.assert_array_equal(bd.birth_positions, self.bd.birth_positions)
#         np.testing.assert_array_equal(bd.death_positions, self.bd.death_positions)

#     @mock.patch("spatialsfs.branchingdiffusion.np.random.default_rng", autospec=True)
#     @mock.patch("spatialsfs.branchingdiffusion.BranchingDiffusion", autospec=True)
#     def test_simulate_branching_diffusions(self, mock_bd, mock_rng):
#         """Test simulation wrapper function."""
#         num_reps = 3
#         s = 0.1
#         d = 0.5
#         ndim = 1
#         rng = np.random.default_rng()
#         max_steps = 5
#         expected_calls = [
#             mock.call(),
#             mock.call().simulate_tree(s, max_steps, rng),
#             mock.call().simulate_positions(d, ndim, rng),
#         ] * num_reps
#         bds = simulate_branching_diffusions(
#             num_reps, s, d, ndim, rng=rng, max_steps=max_steps
#         )
#         self.assertEqual(mock_bd.mock_calls, expected_calls)
#         self.assertEqual(len(bds), num_reps)

#         # Test initialize rng
#         mock_rng.reset_mock()
#         bds = simulate_branching_diffusions(num_reps, s, d, ndim, max_steps=max_steps)
#         mock_rng.assert_called_once()


# def TestIntegration(TestCase):
#     """Test that everything works together."""

#     def test_simulate_branching_diffusions(self):
#         """Test that for some random seeds, everything gets set."""
#         num_reps = 100
#         s = 0.1
#         d = 0.5
#         ndim = 2
#         rng = np.random.default_rng(1)
#         max_steps = 10
#         bds = simulate_branching_diffusions(
#             num_reps, s, d, ndim, rng=rng, max_steps=max_steps
#         )
#         self.assertEqual(len(bds), num_reps)
#         for bd in bds:
#             self.assertEqual(bd.selection_coefficient, s)
#             self.assertEqual(bd.diffusion_coefficient, d)
#             self.assertEqual(bd.ndim, ndim)
#             self.assertEqual(len(bd.parents), bd.num_total)
#             self.assertEqual(bd.birth_times.shape, (bd.num_total,))
#             self.assertEqual(bd.birth_positions.shape, (bd.num_total, ndim))
#             self.assertEqual(bd.death_times.shape, (bd.num_total,))
#             self.assertEqual(bd.death_positions.shape, (bd.num_total, ndim))
#             self.assertEqual(bd.extinction_time, np.max(bd.death_times))

#         with TemporaryFile() as tf:
#             save_branching_diffusions(tf, bds)
#             tf.seek(0)
#             loaded_data = load_branching_diffusions(tf)
#         self.assertEqual(bds, loaded_data)
