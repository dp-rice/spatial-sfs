"""Simulation estimators to use with `montecarloop.Dealer`."""

import numpy as np

from .simulations import branch, simulate_branching_diffusion


class BranchingEstimator:
    """Simulator/Estimator for BranchingProcess."""

    def __init__(self, sim_params):
        self.num_steps = int(sim_params["num_steps"])
        self.omit_steps = int(sim_params.get("omit_steps", 0))
        self.s = sim_params["s"]

    def simulate(self, seed):
        """Return dictionary of estimates from a BranchingProcess simulation.

        Returned dictionary
        -------------------
        ave_time:
            Average time step duration.
        var_time:
            Variance of time step duration.
        ave_alive:
            Average number alive over life-events (birth or death).
        ave_alive_ctime:
            Average number alive over continuous time.
        """
        branchy = branch(self.num_steps, self.s, seed)
        nums = branchy.num_alive()[self.omit_steps :]
        times = np.diff(branchy.life_events())[self.omit_steps :]
        weights = times / np.sum(times)
        return dict(
            ave_alive=np.mean(nums),
            ave_time=np.mean(times),
            var_time=np.var(times),
            ave_alive_ctime=np.sum(nums[:-1] * weights),
        )


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
