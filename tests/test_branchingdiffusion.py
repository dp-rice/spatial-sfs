"""Tests for the branchingdiffusion module."""
from copy import deepcopy
from tempfile import NamedTemporaryFile, TemporaryFile
from unittest import TestCase, main, mock

import numpy as np

from spatialsfs import (
    BranchingDiffusion,
    load_branching_diffusions,
    save_branching_diffusions,
    simulate_branching_diffusions,
)


class TestBranchingDiffusion(TestCase):
    """Test initializing, loading, saving, and equality of BrachingDiffusion class."""

    def setUp(self):
        """Set up a small instance of BranchingDiffusion."""
        self.bd = BranchingDiffusion()
        self.bd.parents = [None, 0, 0]
        self.bd.birth_times = np.array([0.0, 0.5, 0.5])
        self.bd.death_times = np.array([0.5, 1.0, 1.5])
        # 1D positions
        self.bd.ndims = 1
        self.bd.birth_positions = np.array([0.0, 0.3, 0.3]).reshape(3, self.bd.ndims)
        self.bd.death_positions = np.array([0.3, -0.1, 1.3]).reshape(3, self.bd.ndims)
        self.bd.selection_coefficient = 0.05
        self.bd.diffusion_coefficient = 0.75
        self.bd.num_total = 3
        self.bd.num_max = 2
        self.bd.extinction_time = 1.5

    def test_setUp(self):
        """Test that our example has non-default values for all attributes."""
        bd_default = BranchingDiffusion()
        for attr in list(bd_default.__dict__):
            if type(self.bd.__dict__[attr]) is np.ndarray:
                np.testing.assert_raises(
                    AssertionError,
                    np.testing.assert_array_equal,
                    self.bd.__dict__[attr],
                    bd_default.__dict__[attr],
                )
            else:
                self.assertNotEqual(self.bd.__dict__[attr], bd_default.__dict__[attr])

    def test____init__(self):
        """Test that default initialization creates an empty BranchingDiffusion."""
        bd = BranchingDiffusion()
        self.assertEqual(bd.parents, [])
        self.assertEqual(len(bd.birth_times), 0)
        self.assertEqual(len(bd.death_times), 0)
        self.assertEqual(bd.ndims, 0)
        self.assertEqual(len(bd.birth_positions), 0)
        self.assertEqual(len(bd.death_positions), 0)
        self.assertIsNone(bd.selection_coefficient)
        self.assertIsNone(bd.diffusion_coefficient)
        self.assertEqual(bd.num_total, 0)
        self.assertEqual(bd.num_max, 0)
        self.assertIsNone(bd.extinction_time)

    def test___eq__(self):
        """Test equality."""
        # Fresh objects
        bd1 = BranchingDiffusion()
        bd2 = BranchingDiffusion()
        self.assertEqual(bd1, bd2)
        # Objects with some stuff
        bd1 = deepcopy(self.bd)
        bd2 = deepcopy(self.bd)
        self.assertEqual(bd1, bd2)
        # Value mismatches
        bd2.selection_coefficient *= 2.0
        self.assertNotEqual(bd1, bd2)
        bd2.selection_coefficient = bd1.selection_coefficient
        self.assertEqual(bd1, bd2)
        bd2.birth_times[-1] /= 2.0
        self.assertNotEqual(bd1, bd2)
        # Type mismatches
        bd3 = BranchingDiffusion()
        bd3.parents = np.array([])
        self.assertNotEqual(bd1, bd3)
        bd4 = BranchingDiffusion()
        bd4.num_max = np.array(0)
        self.assertNotEqual(bd1, bd4)
        # Not a BranchingDiffusion()
        self.assertNotEqual(bd1, 5)

    def test_saveload(self):
        """Test saving and loading with a temporary file."""
        # Empty bd
        bd_empty = BranchingDiffusion()
        bd1 = BranchingDiffusion()
        with TemporaryFile() as tf:
            bd_empty.save(tf)
            tf.seek(0)
            bd1.load(tf)
        self.assertEqual(bd_empty, bd1)
        # Full bd
        bd2 = BranchingDiffusion()
        with TemporaryFile() as tf:
            self.bd.save(tf)
            tf.seek(0)
            bd2.load(tf)
        self.assertEqual(self.bd, bd2)

    def test_import(self):
        """Test __init__ with an input file."""
        # Empty input
        bd_empty = BranchingDiffusion()
        with TemporaryFile() as tf:
            bd_empty.save(tf)
            tf.seek(0)
            bd1 = BranchingDiffusion(tf)
        self.assertEqual(bd_empty, bd1)
        # Full input
        with TemporaryFile() as tf:
            self.bd.save(tf)
            tf.seek(0)
            bd2 = BranchingDiffusion(tf)
        self.assertEqual(self.bd, bd2)

    def test_import_filename(self):
        """Test __init__ with an input filename string."""
        with NamedTemporaryFile() as tf:
            self.bd.save(tf)
            tf.seek(0)
            bd = BranchingDiffusion(tf.name)
        self.assertEqual(self.bd, bd)

    def test_saveload_branchingdiffusions(self):
        """Test save_branching_diffusions and load_branching_diffusions."""
        bd1 = deepcopy(self.bd)
        bd2 = deepcopy(self.bd)
        bd2.selection_coefficient = 0.12
        saved_data = [bd1, bd2]
        with TemporaryFile() as tf:
            save_branching_diffusions(tf, saved_data)
            tf.seek(0)
            loaded_data = load_branching_diffusions(tf)
        self.assertEqual(saved_data, loaded_data)

    @mock.patch(
        "spatialsfs.branchingdiffusion.simulations.simulate_tree", autospec=True
    )
    def test_simulate_tree(self, mock_sim):
        """Test simulate_tree method (mocking simulation code)."""
        mock_sim.return_value = (
            self.bd.parents,
            self.bd.birth_times,
            self.bd.death_times,
            self.bd.num_max,
        )
        # Run simulations
        bd = BranchingDiffusion()
        rng = np.random.default_rng()
        s = self.bd.selection_coefficient
        max_steps = 5
        bd.simulate_tree(s, max_steps, rng)
        # Assert called simulate_tree with correct params
        mock_sim.assert_called_once_with(s, max_steps, rng)
        self.assertEqual(bd.selection_coefficient, s)
        # Should reset the positions
        self.assertEqual(len(bd.birth_positions), 0)
        self.assertEqual(len(bd.death_positions), 0)
        # Should assign simulation output
        self.assertEqual(bd.parents, self.bd.parents)
        np.testing.assert_array_equal(bd.birth_times, self.bd.birth_times)
        np.testing.assert_array_equal(bd.death_times, self.bd.death_times)
        self.assertEqual(bd.num_total, self.bd.num_total)
        self.assertEqual(bd.num_max, self.bd.num_max)
        self.assertEqual(bd.extinction_time, self.bd.extinction_time)
        self.assertEqual(bd.extinction_time, np.max(bd.death_times))

    @mock.patch(
        "spatialsfs.branchingdiffusion.simulations.simulate_positions", autospec=True
    )
    def test_simulate_positions(self, mock_sim):
        """Test simulate_positions method (mocking simulation code)."""
        mock_sim.return_value = (
            self.bd.birth_positions,
            self.bd.death_positions,
        )
        rng = np.random.default_rng()
        # Set up bd as though we'd run simulate_times
        bd = deepcopy(self.bd)
        bd.diffusion_coefficient = None
        bd.birth_positions = np.array([], dtype=float)
        bd.death_positions = np.array([], dtype=float)
        # Run simulations
        d = self.bd.diffusion_coefficient
        ndims = self.bd.ndims
        bd.simulate_positions(d, ndims, rng)
        # Assert called simulate_positions with correct params
        mock_sim.assert_called_once()
        self.assertEqual(mock_sim.call_args[0][:3], (d, ndims, self.bd.parents))
        intervals = self.bd.death_times - self.bd.birth_times
        np.testing.assert_array_equal(intervals, mock_sim.call_args[0][3])
        self.assertEqual(bd.diffusion_coefficient, d)
        self.assertEqual(bd.ndims, ndims)
        # Should assign simulation output
        np.testing.assert_array_equal(bd.birth_positions, self.bd.birth_positions)
        np.testing.assert_array_equal(bd.death_positions, self.bd.death_positions)

    def test_num_alive_at(self):
        """Test num_alive_at."""
        # Can't get alive at if haven't run simulations.
        bd = BranchingDiffusion()
        with self.assertRaises(RuntimeError):
            bd.num_alive_at(1.0)
        # Birth and death times.
        # self.bd.birth_times = np.array([0.0, 0.5, 0.5])
        # self.bd.death_times = np.array([0.5, 1.0, 1.5])
        self.assertEqual(self.bd.num_alive_at(-1.0), 0)
        # Lifespan includes birth time
        self.assertEqual(self.bd.num_alive_at(0.0), 1)
        self.assertEqual(self.bd.num_alive_at(0.25), 1)
        # Lifespan does not include death time
        self.assertEqual(self.bd.num_alive_at(0.5), 2)
        self.assertEqual(self.bd.num_alive_at(0.75), 2)
        self.assertEqual(self.bd.num_alive_at(1.0), 1)
        self.assertEqual(self.bd.num_alive_at(1.25), 1)
        self.assertEqual(self.bd.num_alive_at(1.5), 0)
        self.assertEqual(self.bd.num_alive_at(10.0), 0)

    def test_positions_at(self):
        """Test positions_at."""
        # Use a round value for mocking gaussians with sd=1
        mock_rng = mock.Mock()
        mock_rng.standard_normal.return_value = 1.0

        # Can't get positions if haven't run position simulations.
        bd = BranchingDiffusion()
        with self.assertRaises(RuntimeError):
            bd.positions_at(1.0, mock_rng)
        bd.birth_times = self.bd.birth_times.copy()
        bd.death_times = self.bd.death_times.copy()
        with self.assertRaises(RuntimeError):
            bd.positions_at(1.0, mock_rng)

        # t < 0.0 should give an array with first dimension == zero.
        self.assertEqual(self.bd.positions_at(-0.5, mock_rng).shape[0], 0)

        # t == 0.0 should give 0.0
        np.testing.assert_array_equal(
            self.bd.positions_at(0.0, mock_rng), np.zeros((1, 1))
        )

        # t == 0.25 should give an interpolation
        t = 0.25
        expected_position = (
            (t * 0.3 / 0.5)
            + np.sqrt(self.bd.diffusion_coefficient * (0.5 - t) * t / 0.5)
        ).reshape((-1, 1))
        np.testing.assert_array_equal(
            self.bd.positions_at(t, mock_rng), expected_position
        )

        # t == 0.6 should be length 2
        t = 0.6
        ep1 = (
            0.3
            + ((t - 0.5) * (-0.4) / 0.5)
            + np.sqrt(self.bd.diffusion_coefficient * (1.0 - t) * (t - 0.5) / 0.5)
        )
        ep2 = (
            0.3
            + ((t - 0.5) * 1.0 / 1.0)
            + np.sqrt(self.bd.diffusion_coefficient * (1.5 - t) * (t - 0.5) / 1.0)
        )
        expected_position = np.vstack([ep1, ep2]).reshape((-1, 1))
        np.testing.assert_array_equal(
            self.bd.positions_at(t, mock_rng), expected_position
        )

    @mock.patch("spatialsfs.branchingdiffusion.np.random.default_rng", autospec=True)
    @mock.patch("spatialsfs.branchingdiffusion.BranchingDiffusion", autospec=True)
    def test_simulate_branching_diffusions(self, mock_bd, mock_rng):
        """Test simulation wrapper function."""
        num_reps = 3
        s = 0.1
        d = 0.5
        ndims = 1
        rng = np.random.default_rng()
        max_steps = 5
        expected_calls = [
            mock.call(),
            mock.call().simulate_tree(s, max_steps, rng),
            mock.call().simulate_positions(d, ndims, rng),
        ] * num_reps
        bds = simulate_branching_diffusions(
            num_reps, s, d, ndims, rng=rng, max_steps=max_steps
        )
        self.assertEqual(mock_bd.mock_calls, expected_calls)
        self.assertEqual(len(bds), num_reps)

        # Test initialize rng
        mock_rng.reset_mock()
        bds = simulate_branching_diffusions(num_reps, s, d, ndims, max_steps=max_steps)
        mock_rng.assert_called_once()


if __name__ == "__main__":
    print("Hello world!")
    main()
