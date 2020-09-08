"""Test spatial-sfs.simulations."""
# Iterate over seeds


from unittest import TestCase, main, mock

import numpy as np

import spatialsfs.simulations as sims


class TestSimulations(TestCase):
    """TestSimulations."""

    def test__step(self):
        """Test _step."""
        s = 0.1

        # Cases: (alive, parent_choice, num_offspring, expected_output)
        cases = [
            ([0], 0, 0),
            ([0], 0, 2),
            ([0, 5], 0, 0),
            ([0, 5], 1, 0),
        ]
        mock_rng = mock.Mock()
        for alive, parent_choice, num_offspring in cases:
            mock_rng.reset_mock()
            mock_rng.standard_exponential.return_value = 1.0
            mock_rng.standard_exponential.return_value = 1.0
            mock_rng.integers.return_value = parent_choice
            mock_rng.choice.return_value = num_offspring
            interval, parent, no = sims._step(alive, s, mock_rng)
            self.assertEqual(interval, 1.0 / len(alive))
            self.assertEqual(parent, alive[parent_choice])
            self.assertEqual(no, num_offspring)
            mock_rng.standard_exponential.assert_called_once()
            mock_rng.integers.assert_called_once_with(len(alive))
            mock_rng.choice.assert_called_once_with(
                [0, 2], p=[(1 + s) / 2, (1 - s) / 2]
            )

    @mock.patch("spatialsfs.simulations._step", autospec=True)
    def test_simulate_tree(self, mock_step):
        """Test simulate_tree."""
        s = 0.1
        max_steps = 3
        rng = np.random.default_rng()

        # Case: die right away
        mock_step.reset_mock()
        mock_step.return_value = (1.0, 0, 0)
        parents, birth_times, death_times, n_max = sims.simulate_tree(s, max_steps, rng)
        mock_step.assert_called_once()
        # mock_step.assert_called_once_with([0], s, rng)
        self.assertEqual(parents, [None])
        np.testing.assert_array_equal(birth_times, np.array([0.0]))
        np.testing.assert_array_equal(death_times, np.array([1.0]))
        self.assertEqual(n_max, 1)

        # Case: birth until max_steps
        mock_step.reset_mock()
        mock_step.side_effect = [(1.0, i, 2) for i in range(max_steps)]
        parents, birth_times, death_times, n_max = sims.simulate_tree(s, max_steps, rng)
        self.assertEqual(parents, [None, 0, 0, 1, 1, 2, 2])
        np.testing.assert_array_equal(
            birth_times, np.array([0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0])
        )
        np.testing.assert_array_equal(
            death_times, np.array([1.0, 2.0, 3.0] + [np.nan] * 4)
        )
        self.assertEqual(n_max, max_steps + 1)

        # Case: birth, death, death
        mock_step.reset_mock()
        mock_step.side_effect = [(1.0, 0, 2), (1.0, 1, 0), (1.0, 2, 0)]
        parents, birth_times, death_times, n_max = sims.simulate_tree(s, max_steps, rng)
        self.assertEqual(parents, [None, 0, 0])
        np.testing.assert_array_equal(birth_times, np.array([0.0, 1.0, 1.0]))
        np.testing.assert_array_equal(death_times, np.array([1.0, 2.0, 3.0]))
        self.assertEqual(n_max, 2)

    def test_simulate_tree_random(self):
        """Test general properties on some random instances."""
        s = 0.01
        max_steps = 10
        rng = np.random.default_rng(1)
        for rep in range(100):
            parents, birth_times, death_times, n_max = sims.simulate_tree(
                s, max_steps, rng
            )
            self.assertIs(parents[0], None)
            n_total = len(parents)
            self.assertEqual(n_total, len(birth_times))
            self.assertEqual(n_total, len(death_times))
            self.assertTrue(n_total <= 2 * max_steps + 1)
            for i in range(n_total - 1):
                self.assertTrue(birth_times[i] <= birth_times[i + 1])
            for i in range(1, n_total):
                self.assertTrue(0 <= parents[i] and parents[i] < i)
                self.assertEqual(birth_times[i], death_times[parents[i]])

    def test_simulate_positions(self):
        """Test simulate_positions."""
        # Check that birth and death positions are the same length as parents
        pass


if __name__ == "__main__":
    main()
