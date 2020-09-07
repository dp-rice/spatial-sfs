"""Simulating and analyzing branching diffusions."""

import io
import zipfile
from typing import BinaryIO, Iterable, List, Optional, Union

import numpy as np

import spatialsfs.simulations as simulations


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
        self.parents: List[int] = []
        self.birth_times: np.ndarray[float] = np.array([], dtype=float)
        self.death_times: np.ndarray[float] = np.array([], dtype=float)
        self.birth_positions: np.ndarray[float] = np.array([], dtype=float)
        self.death_positions: np.ndarray[float] = np.array([], dtype=float)
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

            return all(
                np.all(self.__dict__[k] == other.__dict__[k]) for k in self.__dict__
            )
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
            if value is not None:
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
        self.parents = list(data["parents"])
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
        """Simulate the ancestry tree and tiems.

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
        """simulate_positions.

        Parameters
        ----------
        diffusion_coefficient : float
            diffusion_coefficient

        Returns
        -------
        None

        """
        self.diffusion_coefficient = diffusion_coefficient
        lifespans = self.death_times - self.birth_times
        self.birth_positions, self.death_positions = simulations.simulate_positions(
            self.diffusion_coefficient, self.parents, lifespans
        )

    def num_alive_at(self, time: float) -> int:
        """num_alive_at.

        Parameters
        ----------
        time : float
            time

        Returns
        -------
        int

        """
        pass

    def positions_at(self, time: float) -> np.ndarray[float]:
        """positions_at.

        Parameters
        ----------
        time : float
            time

        Returns
        -------
        np.ndarray[float]

        """
        pass


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
    with zipfile.ZipFile(output_file, "w") as outfile:
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
    with zipfile.ZipFile(input_file, "r") as infile:
        for i in infile.namelist():
            with io.BytesIO() as f:
                f.write(infile.read(str(i)))
                f.seek(0)
                bd_list.append(BranchingDiffusion(f))
    return bd_list
