"""Define common test fixtures."""
import numpy as np
import pytest

from spatialsfs.branchingdiffusion import BranchingDiffusion
from spatialsfs.branchingprocess import BranchingProcess


def pytest_generate_tests(metafunc):
    """Iterate all tests over three dimensions."""
    if "ndim" in metafunc.fixturenames:
        metafunc.parametrize("ndim", [1, 2, 3])


@pytest.fixture
def small_bp():
    """Return a simple BranchingProcess with no restarts."""
    parents = np.array([0, 0, 1, 1])
    birth_times = np.array([0.0, 0.0, 0.5, 0.5])
    death_times = np.array([0.0, 0.5, 0.75, 0.75])
    s = 0.05
    return BranchingProcess(parents, birth_times, death_times, s)


@pytest.fixture
def large_bp():
    """Return a more complex BranchingProcess with one restart."""
    s = 0.05
    return BranchingProcess(
        np.array([0, 0, 1, 1, 0, 4, 4]),
        np.array([0.0, 0.0, 1.0, 1.0, 2.0, 3.5, 3.5]),
        np.array([0.0, 1.0, 1.5, 2.0, 3.5, 4.0, 4.0]),
        s,
    )


@pytest.fixture
def small_bd(small_bp, ndim):
    """Return a simple BranchingDiffusion with no restarts."""
    # ndim = 1
    birth_positions = np.array([0.0, 0.0, 0.25, 0.25] * ndim).reshape((ndim, -1)).T
    death_positions = np.array([0.0, 0.25, 0.35, 0.20] * ndim).reshape((ndim, -1)).T
    d = 0.5
    return BranchingDiffusion(small_bp, birth_positions, death_positions, d)


# @pytest.fixture
# def large_bd():
#     """Return a more complex BranchingProcess with one restart."""
#     s = 0.05
#     return BranchingProcess(
#         np.array([0, 0, 1, 1, 0, 4, 4]),
#         np.array([0.0, 0.0, 1.0, 1.0, 2.0, 3.5, 3.5]),
#         np.array([0.0, 1.0, 1.5, 2.0, 3.5, 4.0, np.inf]),
#         s,
#     )
