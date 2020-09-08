"""Simulating and analyzing branching diffusions."""

import io
import zipfile
from typing import BinaryIO, Iterable, List, Optional, Union

import numpy as np

import spatialsfs.simulations as simulations

# For saving and loading None objects as integers with np.savez_compressed
_ROOT = -1


class BranchingDiffusion:
    """Branching diffusion simulation output object."""

    def __init__(self, input_file: Union[str, BinaryIO, None] = None) -> None:
        """Create a BranchingDiffusion object.

        Parameters
        ----------
        inputfile : Union[str, BinaryIO, None]
            Optional .npz filename or a BinaryIO object to read data from.
            default: construct an empty BranchingDiffusion.
        """
        self.parents: List[Optional[int]] = []
        self.birth_times: np.ndarray = np.array([], dtype=float)
        self.death_times: np.ndarray = np.array([], dtype=float)
        self.birth_positions: np.ndarray = np.array([], dtype=float)
        self.death_positions: np.ndarray = np.array([], dtype=float)
        self.num_total: int = 0
        self.num_max: int = 0
        self.extinction_time: Optional[float] = None
        # Simulation parameters
        self.selection_coefficient: Optional[float] = None
        self.diffusion_coefficient: Optional[float] = None
        if input_file is not None:
            self.load(input_file)

    def __eq__(self, other) -> bool:
        """Return True if all attributes are equal elementwise."""
        if isinstance(other, self.__class__):
            for attr, my_value in self.__dict__.items():
                other_value = other.__dict__[attr]
                if type(my_value) != type(other_value):
                    return False
                if type(my_value) is np.ndarray:
                    if not np.all(my_value == other_value):
                        return False
                elif my_value != other_value:
                    return False
            else:
                return True
        else:
            return NotImplemented

    def save(self, output_file: Union[str, BinaryIO]) -> None:
        """Save BranchingDiffusion to output file.

        Parameters
        ----------
        output_file : Union[str, BinaryIO]
            The file to write the output to.
            May be a filename string or a binary file-like object.

        Returns
        -------
        None

        """
        output_dict = {}
        for key, value in self.__dict__.items():
            # Numpy can't read in None objects without pickle.
            if value is None:
                continue
            elif key == "parents":
                output_dict[key] = [_ROOT if p is None else p for p in value]
            else:
                output_dict[key] = value
        np.savez_compressed(output_file, **output_dict)

    def load(self, input_file: Union[str, BinaryIO]) -> None:
        """Load branching diffusion from file.

        Parameters
        ----------
        input_file : Union[str, BinaryIO]
            The input file in .npz format.
            May be the filename as a string or a BinaryIO object to read data from.
        """
        data = np.load(input_file)
        self.parents = [None if p == _ROOT else p for p in data["parents"]]
        self.birth_times = data["birth_times"]
        self.death_times = data["death_times"]
        self.birth_positions = data["birth_positions"]
        self.death_positions = data["death_positions"]
        self.num_total = int(data["num_total"])
        self.num_max = int(data["num_max"])
        try:
            self.extinction_time = float(data["extinction_time"])
        except KeyError:
            self.extinction_time = None
        try:
            self.selection_coefficient = float(data["selection_coefficient"])
        except KeyError:
            self.selection_coefficient = None
        try:
            self.diffusion_coefficient = float(data["diffusion_coefficient"])
        except KeyError:
            self.diffusion_coefficient = None

    def simulate_tree(
        self, selection_coefficient: float, max_steps: int = 10000
    ) -> None:
        """Simulate the ancestry tree and times.

        Parameters
        ----------
        selection_coefficient : float
            The selection coefficient against the branching process.
            The birth rate is (1-s/2) and death rate is (1+s/2).
        max_steps : int
            The maximum number of steps to take before exiting.
            Default: 10000

        """
        self.selection_coefficient = selection_coefficient
        # New trees will have different numbers of individuals,
        # so we have to reset positions
        self.birth_positions = np.array([], dtype=float)
        self.death_positions = np.array([], dtype=float)
        (
            self.parents,
            self.birth_times,
            self.death_times,
            self.num_max,
        ) = simulations.simulate_tree(self.selection_coefficient, max_steps)
        self.num_total = len(self.parents)
        self.extinction_time = np.max(self.death_times)

    def simulate_positions(self, diffusion_coefficient: float) -> None:
        """Simulate birth and death positions.

        Parameters
        ----------
        diffusion_coefficient : float
            The diffusion coefficient.
        """
        self.diffusion_coefficient = diffusion_coefficient
        lifespans = self.death_times - self.birth_times
        self.birth_positions, self.death_positions = simulations.simulate_positions(
            self.diffusion_coefficient, self.parents, lifespans
        )

    def num_alive_at(self, time: float) -> int:
        """Return the number of individuals alive at a time.

        Parameters
        ----------
        time : float
            Time at which to count the living individuals.

        Returns
        -------
        int
            The number of living individuals at `time`.
            Includes individuals moment they are born,
            but not at the moment they die.

        """
        if self.parents == []:
            raise RuntimeError("Tree not simulated.")
        alive = (self.birth_times <= time) & (time < self.death_times)
        return np.count_nonzero(alive)

    def positions_at(self, time: float) -> np.ndarray:
        """Return random interpolated positions of individuals alive at a time.

        Parameters
        ----------
        time : float
            time

        Returns
        -------
        positions : np.ndarray
            1D array of interpolated positions.
            The length of the array is the number of individuals alive at `time`.
            Includes individuals the moment they are born,
            but not at the moment they die.

        Notes
        -----
        The positions are modeled as a Brownian bridge, conditional on birth and death
        times and locations. Repeated calls to `positions_at` will generate independent
        draws from the Brownian bridge.

        """
        if len(self.birth_positions) == 0:
            raise RuntimeError("Positions not simulated.")
        alive = (self.birth_times <= time) & (time < self.death_times)
        bt = self.birth_times[alive]
        dt = self.death_times[alive]
        bp = self.birth_positions[alive]
        dp = self.death_positions[alive]
        means = bp + (time - bt) * (dp - bp) / (dt - bt)
        variances = (dt - time) * (time - bt) / (dt - bt)
        return means + np.sqrt(variances) * np.random.normal(size=len(bt))


def save_branching_diffusions(
    output_file: Union[str, BinaryIO],
    branching_diffusions: Iterable[BranchingDiffusion],
) -> None:
    """Save a list of branching diffusions to a file.

    Parameters
    ----------
    output_file : Union[str, BinaryIO]
        The output filename or file-like object.
    branching_diffusions : Iterable[BranchingDiffusion]
        A list (or other Iterable) of BranchingDiffusion objects to write.
    """
    with zipfile.ZipFile(output_file, "w", compression=zipfile.ZIP_DEFLATED) as outfile:
        for i, bd in enumerate(branching_diffusions):
            with io.BytesIO() as f:
                bd.save(f)
                f.seek(0)
                outfile.writestr(str(i), f.read())


def load_branching_diffusions(
    input_file: Union[str, BinaryIO]
) -> List[BranchingDiffusion]:
    """Load a list of BranchingDiffusion objects from a file.

    Parameters
    ----------
    input_file : Union[str, BinaryIO]
        The input filename or file-like object.

    Returns
    -------
    List[BranchingDiffusion]
    """
    bd_list = []
    with zipfile.ZipFile(input_file, "r", compression=zipfile.ZIP_DEFLATED) as infile:
        for i in infile.namelist():
            with io.BytesIO() as f:
                f.write(infile.read(str(i)))
                f.seek(0)
                bd_list.append(BranchingDiffusion(f))
    return bd_list


def simulate_branching_diffusions(
    num_reps: int, selection_coefficient: float, diffusion_coefficient: float = 1.0
) -> List[BranchingDiffusion]:
    """Simulate replicate branching diffusions.

    Parameters
    ----------
    num_reps : int
        The number of replicate simulations to run.
    selection_coefficient : float
        The selection coefficient for all simulations.
    diffusion_coefficient : float
        The diffusion coefficient for all simulations.

    Returns
    -------
    List[BranchingDiffusion]

    """
    bds = []
    for i in range(num_reps):
        bd = BranchingDiffusion()
        bd.simulate_tree(selection_coefficient)
        bd.simulate_positions(diffusion_coefficient)
        bds.append(bd)
    return bds
