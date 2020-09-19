"""Test the io module."""

from tempfile import TemporaryFile

import numpy as np
import pytest

from spatialsfs.branchingdiffusion import BranchingDiffusion
from spatialsfs.branchingprocess import BranchingProcess
from spatialsfs.io import (
    load_branching_diffusion,
    load_branching_process,
    save_branching_diffusion,
    save_branching_process,
)


@pytest.fixture
def small_bp():
    """Return a simple BranchingProcess with no restarts."""
    parents = np.array([0, 0, 1, 1])
    birth_times = np.array([0.0, 0.0, 0.5, 0.5])
    death_times = np.array([0.0, 0.5, 0.75, np.inf])
    s = 0.05
    return BranchingProcess(parents, birth_times, death_times, s)


@pytest.fixture
def small_bd(small_bp):
    """Return a simple BranchingDiffusion with no restarts."""
    birth_positions = np.array([0.0, 0.0, 0.25, 0.25]).reshape((4, 1))
    death_positions = np.array([0.0, 0.25, 0.35, np.nan]).reshape((4, 1))
    d = 0.5
    return BranchingDiffusion(small_bp, birth_positions, death_positions, d)


def test_saveload_branching_process(small_bp):
    """Test save and load BranchingProcess."""
    with TemporaryFile() as tf:
        save_branching_process(tf, small_bp)
        tf.seek(0)
        bp = load_branching_process(tf)
    assert bp == small_bp


def test_saveload_branching_diffusion(small_bd):
    """Test save and load BranchingDiffusion."""
    with TemporaryFile() as tf:
        save_branching_diffusion(tf, small_bd)
        tf.seek(0)
        bd = load_branching_diffusion(tf)
    assert bd == small_bd
