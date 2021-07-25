import logging
import math
import multiprocessing as mp
from collections import deque
from multiprocessing.pool import Pool
from typing import Callable, Deque, Optional, Tuple, Union

import numpy as np
from pyswarms.backend.operators import (compute_objective_function,
                                        compute_pbest)
from pyswarms.single import GlobalBestPSO

from ....backend.operators import compute_particle_mean_distances
from .evolution_state import EvolutionState
from .strategy import AccelerationStrategy, BasicAccelerationStrategy
from .apso_options import APSOOptions


class AdaptiveOptimizerPSO(GlobalBestPSO):
    bounds: Tuple[np.ndarray, np.ndarray]  # super allows none but APSO does not
    state: EvolutionState
    acc_strategy: AccelerationStrategy
    apso_options: APSOOptions

    def __init__(
        self,
        n_particles: int,
        dimensions: int,
        apso_options: APSOOptions,
        bounds: Tuple[np.ndarray, np.ndarray],
        acc_strategy: AccelerationStrategy = BasicAccelerationStrategy(),
        bh_strategy: str = "periodic",
        velocity_clamp: Optional[Tuple[float, float]] = None,
        vh_strategy: str = "unmodified",
        center: Union[np.ndarray, float] = 1.00,
        ftol: float = -np.inf,
        ftol_iter: int = 1,
        init_pos: Optional[np.ndarray] = None,
    ):
        """Initialize the swarm

        Attributes
        ----------
        n_particles : int
            number of particles in the swarm.
        dimensions : int
            number of dimensions in the space.
        options : dict with keys :code:`{'c1', 'c2', 'w'}` or :code:`{'c1', 'c2', 'w', 'k', 'p'}`
            a dictionary containing the parameters for the specific
            optimization technique.
                * c1 : float
                    cognitive parameter
                * c2 : float
                    social parameter
                * w : float
                    inertia parameter
                if used with the :code:`Ring`, :code:`VonNeumann` or
                :code:`Random` topology the additional parameter k must be
                included
                * k : int
                    number of neighbors to be considered. Must be a positive
                    integer less than :code:`n_particles`
                if used with the :code:`Ring` topology the additional
                parameters k and p must be included
                * p: int {1,2}
                    the Minkowski p-norm to use. 1 is the sum-of-absolute
                    values (or L1 distance) while 2 is the Euclidean (or L2)
                    distance.
                if used with the :code:`VonNeumann` topology the additional
                parameters p and r must be included
                * r: int
                    the range of the VonNeumann topology.  This is used to
                    determine the number of neighbours in the topology.
        topology : pyswarms.backend.topology.Topology
            a :code:`Topology` object that defines the topology to use in the
            optimization process. The currently available topologies are:
                * Star
                    All particles are connected
                * Ring (static and dynamic)
                    Particles are connected to the k nearest neighbours
                * VonNeumann
                    Particles are connected in a VonNeumann topology
                * Pyramid (static and dynamic)
                    Particles are connected in N-dimensional simplices
                * Random (static and dynamic)
                    Particles are connected to k random particles
                Static variants of the topologies remain with the same
                neighbours over the course of the optimization. Dynamic
                variants calculate new neighbours every time step.
        bounds : tuple of numpy.ndarray
            a tuple of size 2 where the first entry is the minimum bound while
            the second entry is the maximum bound. Each array must be of shape
            :code:`(dimensions,)`.
        bh_strategy : str
            a strategy for the handling of out-of-bounds particles.
        velocity_clamp : tuple, optional
            a tuple of size 2 where the first entry is the minimum velocity and
            the second entry is the maximum velocity. It sets the limits for
            velocity clamping.
        vh_strategy : str
            a strategy for the handling of the velocity of out-of-bounds particles.
        center : numpy.ndarray or float, optional
            controls the mean or center whenever the swarm is generated randomly.
            Default is :code:`1`
        ftol : float
            relative error in objective_func(best_pos) acceptable for
            convergence. Default is :code:`-np.inf`
        ftol_iter : int
            number of iterations over which the relative error in
            objective_func(best_pos) is acceptable for convergence.
            Default is :code:`1`
        init_pos : numpy.ndarray, optional
            option to explicitly set the particles' initial positions. Set to
            :code:`None` if you wish to generate the particles randomly.
        """
        super(AdaptiveOptimizerPSO, self).__init__(
            n_particles=n_particles,
            dimensions=dimensions,
            options=apso_options.create_options_dict(),
            bounds=bounds,
            oh_strategy=None,
            bh_strategy=bh_strategy,
            velocity_clamp=velocity_clamp,
            vh_strategy=vh_strategy,
            center=center,
            ftol=ftol,
            ftol_iter=ftol_iter,
            init_pos=init_pos,
        )
        self.acc_strategy = acc_strategy
        self.apso_options = apso_options

    def optimize(self,
                 objective_func: Callable[[np.ndarray], np.ndarray],
                 iters: int,
                 n_processes: Optional[int] = None,
                 verbose: bool = True,
                 **kwargs) -> Tuple[float, np.ndarray]:
        """Optimize the swarm for a number of iterations

        Performs the optimization to evaluate the objective
        function :code:`f` for a number of iterations :code:`iter.`

        Parameters
        ----------
        objective_func : Callable[[np.ndarray],np.ndarray]
            objective function to be evaluated. takes an ndarray of shape (n_particles, n_dimensions) and returns an ndarray (n_particles, )
        iters : int
            number of iterations/generations
        n_processes : int
            number of processes to use for parallel particle evaluation (default: None = no parallelization)
        verbose : bool
            enable or disable the logs and progress bar (default: True = enable logs)
        kwargs : dict
            arguments for the objective function

        Returns
        -------
        tuple
            the global best cost and the global best position.
        """
        # Apply verbosity
        log_level: int
        if verbose:
            log_level = logging.INFO
        else:
            log_level = logging.NOTSET

        self.rep.log("Obj. func. args: {}".format(kwargs), lvl=logging.DEBUG)
        self.rep.log(
            "Optimize for {} iters with {}".format(iters, self.options),
            lvl=log_level,
        )

        # Populate memory of the handlers
        self.bh.memory = self.swarm.position
        self.vh.memory = self.swarm.position

        # Setup Pool of processes for parallel evaluation
        pool: Union[Pool, None] = (None if n_processes is None else
                                   mp.Pool(n_processes))

        self.swarm.pbest_cost = np.full(self.swarm_size[0], np.inf)
        ftol_history: Deque = deque(maxlen=self.ftol_iter)
        for i in self.rep.pbar(iters, self.name) if verbose else range(iters):
            # Compute cost for current position and personal best
            # fmt: off
            self.swarm.current_cost = compute_objective_function(self.swarm,
                                                                 objective_func,
                                                                 pool=pool,
                                                                 **kwargs)
            self.swarm.pbest_pos, self.swarm.pbest_cost = compute_pbest(
                self.swarm)
            best_cost_yet_found = self.swarm.best_cost
            # fmt: on
            # Update swarm
            self.swarm.best_pos, self.swarm.best_cost = self.top.compute_gbest(
                self.swarm, **self.options)
            # Print to console
            if verbose:
                self.rep.hook(best_cost=self.swarm.best_cost)
            hist = self.ToHistory(
                best_cost=self.swarm.best_cost,
                mean_pbest_cost=np.mean(self.swarm.pbest_cost),
                mean_neighbor_cost=self.swarm.best_cost,
                position=self.swarm.position,
                velocity=self.swarm.velocity,
            )
            self._populate_history(hist)
            self.__perform_ese()
            # Verify stop criteria based on the relative acceptable cost ftol
            relative_measure = self.ftol * (1 + np.abs(best_cost_yet_found))
            delta = (np.abs(self.swarm.best_cost - best_cost_yet_found) <
                     relative_measure)
            if i < self.ftol_iter:
                ftol_history.append(delta)
            else:
                ftol_history.append(delta)
                if all(ftol_history):
                    break
            # Perform velocity and position updates
            self.swarm.velocity = self.top.compute_velocity(
                self.swarm, self.velocity_clamp, self.vh, self.bounds)
            self.swarm.position = self.top.compute_position(
                self.swarm, self.bounds, self.bh)
            if i != iters - 1:
                self.__perform_els(objective_func=objective_func,
                                   curr_iter=i,
                                   max_iter=iters)

        # Obtain the final best_cost and the final best_position
        final_best_cost: float = self.swarm.best_cost.copy()
        final_best_pos: np.ndarray = self.swarm.pbest_pos[
            self.swarm.pbest_cost.argmin()].copy()
        # Write report in log and return final cost and position
        self.rep.log(
            "Optimization finished | best cost: {}, best pos: {}".format(
                final_best_cost, final_best_pos),
            lvl=log_level,
        )
        # Close Pool of Processes
        if n_processes is not None:
            pool.close()
        return (final_best_cost, final_best_pos)

    def reset(self):
        super().reset()
        self.state = EvolutionState.S1_EXPLORATION

    def __perform_ese(self):
        mean_distances: np.ndarray = compute_particle_mean_distances(self.swarm)
        evo_factor: float = self.__compute_evolutionary_factor(
            mean_distances=mean_distances)
        state_numeric: np.ndarray = self.__compute_state_numeric(
            evo_factor=evo_factor)
        self.state = EvolutionState.classify_next_state(
            state_numeric=state_numeric, curr_state=self.state)
        self.__update_intertia(evo_factor=evo_factor)
        self.__update_acceleration()

    def __compute_evolutionary_factor(self,
                                      mean_distances: np.ndarray) -> float:
        """Calculate the evolutionary factor.

        The evolutionary factor f = (d_g - d_min)/(d_max - d_min) is in range [0, 1], where d_g is the mean distance of the current global best, d_min and d_max is the maximum and minimum mean distance in mean_distances, respectively.

        Parameters
        ----------
        swarm : Swarm
            the swarm to be evaluated
        mean_distances : np.ndarray
            (n_particles, ) the mean distances of all particles.

        Returns
        -------
        float
            the evolutionary factor.
        """
        min_mean_dist: float = np.min(mean_distances)
        max_mean_dist: float = np.max(mean_distances)
        gbest_idx: int = np.argmin(self.swarm.current_cost)
        gbest_mean_dist: float = mean_distances[gbest_idx]
        evo_factor = (gbest_mean_dist - min_mean_dist) / (max_mean_dist -
                                                          min_mean_dist)
        return evo_factor

    def __compute_state_numeric(self, evo_factor: float) -> np.ndarray:
        """Compute evolutionary state numeric.

        Parameters
        ----------
        evo_factor : float
            the current evolutionary factor

        Returns
        -------
        np.ndarray
            (4,) in the order of [s1_num, s2_num, s3_num, s4_num].
        """
        state_num: np.ndarray = np.full(shape=(4), fill_value=0.0)
        state_num[0] = EvolutionState.compute_exploration_numeric(
            evo_factor=evo_factor)
        state_num[1] = EvolutionState.compute_exploitation_numeric(
            evo_factor=evo_factor)
        state_num[2] = EvolutionState.compute_convergence_numeric(
            evo_factor=evo_factor)
        state_num[3] = EvolutionState.compute_jump_out_numeric(
            evo_factor=evo_factor)
        return state_num

    def __update_intertia(self, evo_factor: float) -> None:
        """Update inertia.

        Update the inertia according to: w(f) = 1/(1+1.5e^(-2.6*evo_factor)).

        Parameters
        ----------
        evo_factor : float
            the evolutionary factor at current step
        """
        new_inertia: float = 1.0 / (1.0 + 1.5 * math.exp(-2.6 * evo_factor))
        self.swarm.options["w"] = new_inertia
        if self.swarm.options is not self.options:
            self.options["w"] = new_inertia

    def __update_acceleration(self) -> None:
        if self.state is EvolutionState.S1_EXPLORATION:
            self.options["c1"], self.options[
                "c2"] = self.acc_strategy.exploration_strategy(
                    self.options["c1"], self.options["c2"])
        elif self.state is EvolutionState.S2_EXPLOITATION:
            self.options["c1"], self.options[
                "c2"] = self.acc_strategy.exploitation_strategy(
                    self.options["c1"], self.options["c2"])
        elif self.state is EvolutionState.S3_CONVERGENCE:
            self.options["c1"], self.options[
                "c2"] = self.acc_strategy.convergence_strategy(
                    self.options["c1"], self.options["c2"])
        elif self.state is EvolutionState.S4_JUMPING_OUT:
            self.options["c1"], self.options[
                "c2"] = self.acc_strategy.jumping_out_strategy(
                    self.options["c1"], self.options["c2"])

    def __compute_sigma(self, curr_iter: int, max_iter: int) -> float:
        sigma_max: float = self.apso_options.sigma_max
        sigma_min: float = self.apso_options.sigma_min
        sigma: float = sigma_max - (sigma_max - sigma_min) * (curr_iter /
                                                              max_iter)
        return sigma

    def __perform_els(self, objective_func: Callable[[np.ndarray], np.ndarray],
                      curr_iter: int, max_iter: int):
        mod_dim: int = int(np.random.uniform(1, self.dimensions))
        mod_pos: np.ndarray = self.swarm.best_pos.copy()
        x_min_d: float = self.bounds[0][mod_dim]
        x_max_d: float = self.bounds[1][mod_dim]
        sigma = self.__compute_sigma(curr_iter=curr_iter, max_iter=max_iter)
        mod_pos[mod_dim] = mod_pos[mod_dim] + (
            x_max_d - x_min_d) * np.random.normal(loc=0.0, scale=sigma)
        mod_pos = self.bh(position=np.expand_dims(a=mod_pos, axis=0),
                          bounds=self.bounds)[0]
        mod_pos_cost: float = objective_func(np.expand_dims(a=mod_pos,
                                                            axis=0))[0]
        if mod_pos_cost < self.swarm.best_cost:
            best_idx: int = np.argmin(self.swarm.current_cost)
            self.swarm.best_cost = mod_pos_cost
            self.swarm.best_pos = mod_pos
            self.swarm.current_cost[best_idx] = mod_pos_cost
            self.swarm.position[best_idx] = mod_pos
        else:
            worst_idx: int = np.argmin(self.swarm.current_cost)
            self.swarm.current_cost[worst_idx] = mod_pos_cost
            self.swarm.position[worst_idx] = mod_pos
