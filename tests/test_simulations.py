"""Test spatial-sfs.simulations."""
# Iterate over seeds


from unittest import TestCase, main, mock

import spatialsfs.simulations as sims


class TestSimulations(TestCase):
    """TestSimulations."""

    def test__step(self):
        """Test __step."""
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

    def test_simulate_tree(self):
        """Test simulate_tree."""
        # Mock patch _step

        # Case: die right away

        # Case: birth until max_steps

        # Case: birth, birth, death

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


if __name__ == "__main__":
    main()
