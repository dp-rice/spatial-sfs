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

        # Cases: (alive, parent_choice, num_offspring)
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
            mock_rng.integers.return_value = parent_choice
            mock_rng.binomial.return_value = num_offspring // 2
            interval, parent, no = sims._step(alive, s, mock_rng)
            self.assertEqual(interval, 1.0 / len(alive))
            self.assertEqual(parent, alive[parent_choice])
            self.assertEqual(no, num_offspring)
            mock_rng.standard_exponential.assert_called_once()
            mock_rng.integers.assert_called_once_with(len(alive))

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


if __name__ == "__main__":
    main()
