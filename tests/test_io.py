"""Test the io module."""


def test_save():
    """TODO."""
    pass


def test_load():
    """TODO."""
    pass


#     def test_saveload(self):
#         """Test saving and loading with a temporary file."""
#         # Empty bd
#         bd_empty = BranchingDiffusion()
#         bd1 = BranchingDiffusion()
#         with TemporaryFile() as tf:
#             bd_empty.save(tf)
#             tf.seek(0)
#             bd1.load(tf)
#         self.assertEqual(bd_empty, bd1)
#         # Full bd
#         bd2 = BranchingDiffusion()
#         with TemporaryFile() as tf:
#             self.bd.save(tf)
#             tf.seek(0)
#             bd2.load(tf)
#         self.assertEqual(self.bd, bd2)

#     def test_import(self):
#         """Test __init__ with an input file."""
#         # Empty input
#         bd_empty = BranchingDiffusion()
#         with TemporaryFile() as tf:
#             bd_empty.save(tf)
#             tf.seek(0)
#             bd1 = BranchingDiffusion(tf)
#         self.assertEqual(bd_empty, bd1)
#         # Full input
#         with TemporaryFile() as tf:
#             self.bd.save(tf)
#             tf.seek(0)
#             bd2 = BranchingDiffusion(tf)
#         self.assertEqual(self.bd, bd2)

#     def test_import_filename(self):
#         """Test __init__ with an input filename string."""
#         with NamedTemporaryFile() as tf:
#             self.bd.save(tf)
#             tf.seek(0)
#             bd = BranchingDiffusion(tf.name)
#         self.assertEqual(self.bd, bd)


# # def test_saveload_branchingdiffusions(self):
# #         """Test save_branching_diffusions and load_branching_diffusions."""
# #         bd1 = deepcopy(self.bd)
# #         bd2 = deepcopy(self.bd)
# #         bd2.selection_coefficient = 0.12
# #         saved_data = [bd1, bd2]
# #         with TemporaryFile() as tf:
# #             save_branching_diffusions(tf, saved_data)
# #             tf.seek(0)
# #             loaded_data = load_branching_diffusions(tf)
# #         self.assertEqual(saved_data, loaded_data)

#         with TemporaryFile() as tf:
#             save_branching_diffusions(tf, bds)
#             tf.seek(0)
#             loaded_data = load_branching_diffusions(tf)
#         self.assertEqual(bds, loaded_data)
