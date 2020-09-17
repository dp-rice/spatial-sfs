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


# def brownian_bridge(
#     t: float,
#     t_a: np.array,
#     t_b: np.array,
#     x_a: np.array,
#     x_b: np.array,
#     diffusion_coefficient: float,
#     seed: int,
# ) -> np.array:
#     """Return random positions drawn from n independent Brownian bridges.

#     Parameters
#     ----------
#     t : float
#         The time at which to sample the Brownian bridges.
#     t_a : np.array
#         1D array with the initial times of the bridges.
#         Shape is (n,)
#     t_b : np.array
#         1D array with the final times of the bridges.
#         Shape is (n,)
#     x_a : np.array
#         2D array with the initial positions of the bridges.
#         Shape is (n, ndims)
#     x_b : np.array
#         2D array with the initial positions of the bridges.
#         Shape is (n, ndims)
#     diffusion_coefficient : float
#         The diffusion coefficient of the brownian bridge.
#     seed : int
#         A seed for numpy random generator.

#     Returns
#     -------
#     np.array
#         The positions at t. Shape is (n, ndims).
#     """
#     rng = np.random.default_rng(seed)
#     means = x_a + (x_b - x_a) * ((t - t_a) / (t_b - t_a))[:, None]
#     variances = diffusion_coefficient * (t_b - t) * (t - t_a) / (t_b - t_a)
#     return means + np.sqrt(variances)[:, None] * rng.standard_normal(size=x_a.shape)
