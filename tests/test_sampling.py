"""Test spatialsfs.sampling."""

# def test_positions_at(small_bd):
#     """Test positions_at."""
#     # TODO: write some actual tests
#     # Placeholder for when I really test it.
#     small_bd.positions_at(0.25, 100)


#     def test_positions_at(self):
#         """Test positions_at."""
#         # Use a round value for mocking gaussians with sd=1
#         mock_rng = mock.Mock()
#         mock_rng.standard_normal.return_value = 1.0

#         # t < 0.0 should give an array with first dimension == zero.
#         self.assertEqual(self.bd.positions_at(-0.5, mock_rng).shape[0], 0)

#         # t == 0.0 should give 0.0
#         np.testing.assert_array_equal(
#             self.bd.positions_at(0.0, mock_rng), np.zeros((1, 1))
#         )

#         # t == 0.25 should give an interpolation
#         t = 0.25
#         expected_position = (
#             (t * 0.3 / 0.5)
#             + np.sqrt(self.bd.diffusion_coefficient * (0.5 - t) * t / 0.5)
#         ).reshape((-1, 1))
#         np.testing.assert_array_equal(
#             self.bd.positions_at(t, mock_rng), expected_position
#         )

#         # t == 0.6 should be length 2
#         t = 0.6
#         ep1 = (
#             0.3
#             + ((t - 0.5) * (-0.4) / 0.5)
#             + np.sqrt(self.bd.diffusion_coefficient * (1.0 - t) * (t - 0.5) / 0.5)
#         )
#         ep2 = (
#             0.3
#             + ((t - 0.5) * 1.0 / 1.0)
#             + np.sqrt(self.bd.diffusion_coefficient * (1.5 - t) * (t - 0.5) / 1.0)
#         )
#         expected_position = np.vstack([ep1, ep2]).reshape((-1, 1))
#         np.testing.assert_array_equal(
#             self.bd.positions_at(t, mock_rng), expected_position
#         )
