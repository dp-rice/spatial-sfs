"""Simulation estimators to use with `montecarlo.Dealer`."""

import numpy as np

from spatialsfs import simulate_branching_diffusion


class DiffusionEstimator:
    """Simulator/Estimator for BranchingDiffusion."""

    def __init__(self, sim_params):
        self.nstep = int(sim_params["nstep"])
        self.ndim = int(sim_params.get("ndim", 2))
        self.s = sim_params["s"]
        self.diffusion_coefficient = sim_params["diffusion"]

    def simulate(self, seed):
        """Return dictionary of estimates from a BranchingDiffusion simulation.

        Returned dictionary
        -------------------
        ave_time_adj_coord:
            Average square-root-of-time normalized coordinate values.
            Coordinates across all dimensions are averaged together.
        var_time_adj_dist:
            Variance of square-root-of-time normalized distance of position (from zero).

        """
        result = simulate_branching_diffusion(
            self.nstep, self.s, self.ndim, self.diffusion_coefficient, seed
        )
        restarts = result.branching_process.restarts
        # pick one of the individuals who have traveled far prior to extinction
        # it's ok that it might not exactly be the farthest one traveled
        picks = np.append(restarts[1:], self.nstep) - 1
        times = result.branching_process.death_times[picks]
        times -= result.branching_process.birth_times[restarts]
        good = times > 0.0
        picks = picks[good]
        times = times[good]
        # scale position by sqrt of time for uniform (expected) variance
        z = result.death_positions[picks] / np.sqrt(times)[:, np.newaxis]
        sumsqs = (z ** 2).sum(axis=1)
        assert sumsqs.shape == (len(times),)
        return dict(ave_time_adj_coord=np.mean(z), var_time_adj_dist=np.mean(sumsqs),)
