"""Test spatial-sfs.simulations."""
# Iterate over seeds

# Check that birth times, death times, and parents have same length
# Check that the selection coefficient must be positive
# Check that birth times are in order
# Check that parents are before children
# Check that ancestor birth time is zero and parent is None

# Check that birth and death positions are the same length as parents
from unittest import TestCase, main


class TestSimulations(TestCase):
    """TestSimulations."""

    def test_simulate_tree(self):
        """Test simulate_tree."""
        pass

    def test_simulate_positions(self):
        """Test simulate_positions."""
        pass


if __name__ == "__main__":
    main()
