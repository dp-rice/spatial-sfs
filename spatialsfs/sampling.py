"""Functions for sampling positions and the SFS."""

# def positions_at(self, time: float, seed: int) -> np.ndarray:
#     """Return random interpolated positions of individuals alive at a time.

#     Parameters
#     ----------
#     time : float
#         time

#     Returns
#     -------
#     positions : np.ndarray
#         1D array of interpolated positions.
#         The length of the array is the number of individuals alive at `time`.
#         Includes individuals the moment they are born,
#         but not at the moment they die.
#     seed : int
#         A seed for numpy random generator.

#     Notes
#     -----
#     The positions are modeled as a Brownian bridge, conditional on birth and death
#     times and locations. Repeated calls to `positions_at` will generate independent
#     draws from the Brownian bridge.

#     """
#     if len(self.birth_positions) == 0:
#         raise RuntimeError("Positions not simulated.")
#     if self.diffusion_coefficient is None:
#         raise RuntimeError("Diffusion coefficient not set.")
#     return brownian_bridge(
#         time, *self[self.alive_at(time)], self.diffusion_coefficient, seed
#     )
