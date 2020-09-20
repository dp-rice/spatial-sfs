"""Test the io module."""

from tempfile import TemporaryFile

from spatialsfs.io import (
    load_branching_diffusion,
    load_branching_process,
    save_branching_diffusion,
    save_branching_process,
)


def test_saveload_branching_process(small_bp, large_bp):
    """Test save and load BranchingProcess."""
    with TemporaryFile() as tf:
        save_branching_process(tf, small_bp)
        tf.seek(0)
        bp = load_branching_process(tf)
    assert bp == small_bp

    with TemporaryFile() as tf:
        save_branching_process(tf, large_bp)
        tf.seek(0)
        bp = load_branching_process(tf)
    assert bp == large_bp


def test_saveload_branching_diffusion(small_bd):
    """Test save and load BranchingDiffusion."""
    with TemporaryFile() as tf:
        save_branching_diffusion(tf, small_bd)
        tf.seek(0)
        bd = load_branching_diffusion(tf)
    assert bd == small_bd
