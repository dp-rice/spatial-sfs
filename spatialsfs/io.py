"""Saving and loading BranchingProcess and BranchingDiffusion objects."""

from typing import BinaryIO, Union

import numpy as np

from spatialsfs.branchingdiffusion import BranchingDiffusion
from spatialsfs.branchingprocess import BranchingProcess


def save_branching_process(
    f: Union[str, BinaryIO], branching_process: BranchingProcess
):
    """Save object to file."""
    np.savez_compressed(f, **_bp_dict(branching_process))


def save_branching_diffusion(
    f: Union[str, BinaryIO], branching_diffusion: BranchingDiffusion
):
    """Save object to file."""
    np.savez_compressed(f, **_bd_dict(branching_diffusion))


def _bp_dict(bp):
    return {
        "parents": bp.parents,
        "birth_times": bp.birth_times,
        "death_times": bp.death_times,
        "selection_coefficient": np.array([bp.selection_coefficient], dtype=float),
    }


def _bd_dict(bd):
    return {
        **_bp_dict(bd.branching_process),
        "birth_positions": bd.birth_positions,
        "death_positions": bd.death_positions,
        "diffusion_coefficient": np.array([bd.diffusion_coefficient]),
    }


def load_branching_process(f: Union[str, BinaryIO]) -> BranchingProcess:
    """Load BranchingProcess from file."""
    d = np.load(f)
    return BranchingProcess(
        d["parents"], d["birth_times"], d["death_times"], d["selection_coefficient"][0],
    )


def load_branching_diffusion(f: Union[str, BinaryIO]) -> BranchingDiffusion:
    """Load BranchingDiffusion from file."""
    d = np.load(f)
    return BranchingDiffusion(
        BranchingProcess(
            d["parents"],
            d["birth_times"],
            d["death_times"],
            d["selection_coefficient"][0],
        ),
        d["birth_positions"],
        d["death_positions"],
        d["diffusion_coefficient"][0],
    )
