"""Tests for the branchingdiffusion module."""
from copy import deepcopy
from tempfile import NamedTemporaryFile, TemporaryFile
from unittest import TestCase, main

import numpy as np

from spatialsfs import (
    BranchingDiffusion,
    load_branching_diffusions,
    save_branching_diffusions,
)


class TestBranchingDiffusion(TestCase):
    """Test initializing, loading, saving, and equality of BrachingDiffusion class."""

    def setUp(self):
        """Set up a small instance of BranchingDiffusion."""
        self.bd = BranchingDiffusion()
        self.bd.parents = [-1, 0, 0]
        self.bd.birth_times = np.array([0.0, 0.5, 0.5])
        self.bd.death_times = np.array([0.5, 1.0, 1.5])
        self.bd.birth_positions = np.array([0.0, 0.3, 0.3])
        self.bd.death_positions = np.array([0.3, -0.1, 1.3])
        self.bd.selection_coefficient = 0.05
        self.bd.diffusion_coefficient = 0.75
        self.bd.num_total = 3
        self.bd.num_max = 2
        self.bd.extinction_time = 1.5

    def test____init__(self):
        """Test that default initialization creates an empty BranchingDiffusion."""
        bd = BranchingDiffusion()
        self.assertEqual(bd.parents, [])
        self.assertEqual(len(bd.birth_times), 0)
        self.assertEqual(len(bd.death_times), 0)
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
        # Value mismatch
        bd2.selection_coefficient *= 2.0
        self.assertNotEqual(bd1, bd2)
        # Type mismatches
        bd3 = BranchingDiffusion()
        bd3.parents = np.array([])
        self.assertNotEqual(bd1, bd3)
        bd4 = BranchingDiffusion()
        bd4.num_max = np.array(0)
        self.assertNotEqual(bd1, bd4)

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


class TestSimulations(TestCase):
    """TestSimulations."""

    def setUp(self):
        """Set up."""
        # Run simulations
        pass

    # Try to push the small value of s without breaking things.

    def test_simulate_tree(self):
        """Test simulate_tree."""
        # Iterate over seeds
        # Check that birth times, death times, and parents have same length
        # Check that the selection coefficient must be positive
        # Check that birth times are in order
        # Check that parents are before children
        # Check that ancestor birth time is zero and parent is None
        pass

    def test_simulate_positions(self):
        """Test simulate_positions."""
        # Check that birth and death positions are the same length as parents
        pass


class TestAnalysis(TestCase):
    """TestAnalysis."""

    def test_num_alive_at(self):
        """Test num_alive_at."""
        pass

    def test_positions_at(self):
        """Test positions_at."""
        pass


if __name__ == "__main__":
    main()
