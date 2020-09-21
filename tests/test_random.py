"""Test _random.py module."""
import numpy as np
import pytest

from spatialsfs import _random


@pytest.mark.parametrize("seed", [1, 3931048, np.random.SeedSequence(50)])
def test_seed_handler(seed):
    """Test seed_handler."""
    seedseq = _random.seed_handler(seed)
    if isinstance(seed, np.random.SeedSequence):
        assert seedseq is seed
    else:
        assert seedseq.entropy == seed
