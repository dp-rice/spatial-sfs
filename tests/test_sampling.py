"""Test spatialsfs.sampling."""
import numpy as np
import pytest
from scipy.stats import vonmises

from spatialsfs.sampling import (
    _generate_sample_positions,
    sample,
    sample_intensity,
    sample_positions,
    two_sample_intensity,
)


def test_sample_positions(small_bd):
    """Test sample_positions."""
    seed = 100
    # Should give empty array when sampled too early or late
    assert len(sample_positions(small_bd, -1.0, seed)) == 0
    with pytest.raises(ValueError):
        sample_positions(small_bd, small_bd.branching_process.final_time, seed)
    with pytest.raises(ValueError):
        sample_positions(small_bd, small_bd.branching_process.final_time + 1.0, seed)
    # Test correct shape
    assert sample_positions(small_bd, 0.0, seed).shape == (1, small_bd.ndim)
    assert sample_positions(small_bd, 0.25, seed).shape == (1, small_bd.ndim)
    assert sample_positions(small_bd, 0.50, seed).shape == (2, small_bd.ndim)
    assert sample_positions(small_bd, 0.65, seed).shape == (2, small_bd.ndim)


@pytest.mark.parametrize("kappa", [0.01, 1, 100.0])
def test_sample_intensity(ndim, kappa):
    """Test sample_intensity."""
    x = np.arange(12.0).reshape(-1, ndim)
    L = 5.0
    assert np.isclose(
        sample_intensity(kappa, L, x),
        np.sum(np.product(vonmises(kappa, scale=L / (2 * np.pi)).pdf(x), axis=1)),
    )


@pytest.mark.parametrize("separation", [0.01, 1, 100.0])
def test_two_sample_intensity(ndim, separation):
    """Test symmetry of two_sample_intensity."""
    x = np.zeros((1, ndim))
    kappa = 1.0
    L = 100.0
    int1, int2 = two_sample_intensity(kappa, L, x, separation)
    assert int1 == int2


def test_generate_sample_positions(small_bd):
    """Test generate_sample_positions iterator."""
    times = [0.0, 0.25, 0.5]
    seed = 1
    x_0 = np.ones((len(times), small_bd.ndim))
    gen = _generate_sample_positions(small_bd, times, x_0, seed)
    # Should yield the same random values as the internal rng of gen
    seedseq = np.random.SeedSequence(seed)
    for t in times:
        assert np.all(next(gen) == sample_positions(small_bd, t, seedseq) + 1.0)
    with pytest.raises(StopIteration):
        next(gen)


@pytest.mark.parametrize("num_centers", [1, 2, 3])
def test_sample(small_bd, num_centers):
    """Test that sample runs and returns the correct shape."""
    num_samples = 4
    if num_centers > 2:
        with pytest.raises(ValueError):
            sample(small_bd, num_samples, 2.0, 100.0, 1, num_centers, 1.0)
    else:
        intensities, weights = sample(
            small_bd, num_samples, 2.0, 100.0, 1, num_centers, 1.0
        )
        assert intensities.shape == (num_samples, num_centers)
        assert weights.shape == (num_samples,)
