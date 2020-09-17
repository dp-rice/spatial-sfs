"""Saving and loading BranchingProcess and BranchingDiffusion objects."""


def save():
    """Save object to file."""
    pass


def load():
    """Load object from file."""
    pass


#     def save(self, output_file: Union[str, BinaryIO]) -> None:
#         """Save BranchingDiffusion to output file.

#         Parameters
#         ----------
#         output_file : Union[str, BinaryIO]
#             The file to write the output to.
#             May be a filename string or a binary file-like object.

#         Returns
#         -------
#         None

#         """
#         output_dict = {}
#         for key, value in self.__dict__.items():
#             # Numpy can't read in None objects without pickle.
#             if value is None:
#                 continue
#             elif key == "parents":
#                 output_dict[key] = [_ROOT if p is None else p for p in value]
#             else:
#                 output_dict[key] = value
#         np.savez_compressed(output_file, **output_dict)

#     def load(self, input_file: Union[str, BinaryIO]) -> None:
#         """Load branching diffusion from file.

#         Parameters
#         ----------
#         input_file : Union[str, BinaryIO]
#             The input file in .npz format.
#             May be the filename as a string or a BinaryIO object to read data from.
#         """
#         data = np.load(input_file)
#         self.parents = [None if p == _ROOT else p for p in data["parents"]]
#         self.birth_times = data["birth_times"]
#         self.death_times = data["death_times"]
#         self.ndim = int(data["ndim"])
#         self.birth_positions = data["birth_positions"]
#         self.death_positions = data["death_positions"]
#         self.num_total = int(data["num_total"])
#         self.num_max = int(data["num_max"])
#         try:
#             self.extinction_time = float(data["extinction_time"])
#         except KeyError:
#             self.extinction_time = None
#         try:
#             self.selection_coefficient = float(data["selection_coefficient"])
#         except KeyError:
#             self.selection_coefficient = None
#         try:
#             self.diffusion_coefficient = float(data["diffusion_coefficient"])
#         except KeyError:
#             self.diffusion_coefficient = None

# def save_branching_diffusions(
#     output_file: Union[str, BinaryIO],
#     branching_diffusions: Iterable[BranchingDiffusion],
# ) -> None:
#     """Save a list of branching diffusions to a file.

#     Parameters
#     ----------
#     output_file : Union[str, BinaryIO]
#         The output filename or file-like object.
#     branching_diffusions : Iterable[BranchingDiffusion]
#         A list (or other Iterable) of BranchingDiffusion objects to write.
#     """
#   with zipfile.ZipFile(output_file, "w", compression=zipfile.ZIP_DEFLATED) as outfile:
#         for i, bd in enumerate(branching_diffusions):
#             with io.BytesIO() as f:
#                 bd.save(f)
#                 f.seek(0)
#                 outfile.writestr(str(i), f.read())


# def load_branching_diffusions(
#     input_file: Union[str, BinaryIO]
# ) -> List[BranchingDiffusion]:
#     """Load a list of BranchingDiffusion objects from a file.

#     Parameters
#     ----------
#     input_file : Union[str, BinaryIO]
#         The input filename or file-like object.

#     Returns
#     -------
#     List[BranchingDiffusion]
#     """
#     bd_list = []
#     with zipfile.ZipFile(input_file, "r", compression=zipfile.ZIP_DEFLATED) as infile:
#         for i in infile.namelist():
#             with io.BytesIO() as f:
#                 f.write(infile.read(str(i)))
#                 f.seek(0)
#                 bd_list.append(BranchingDiffusion(f))
#     return bd_list
