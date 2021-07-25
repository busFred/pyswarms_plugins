# -*- coding: utf-8 -*-
"""
Swarm Operation Backend

This module abstracts various operations in the swarm such as updating
the personal best, finding neighbors, etc. You can use these methods
to specify how the swarm will behave.
"""

import logging

import numpy as np
from pyswarms.backend import Swarm
from pyswarms.utils.reporter import Reporter

rep = Reporter(logger=logging.getLogger(__name__))

__all__ = ["compute_particle_mean_distances"]


def compute_particle_mean_distances(swarm: Swarm) -> np.ndarray:
    """Calculate mean distances for all of all particles.

    At the current position, caluclate the mean euclidiain distance of each particle i to all the other particles.

    Parameters
    ----------
    swarm : Swarm
        the swarm to be used to calculate mean distances.

    Returns
    -------
    np.ndarray
        (n_particles, ) mean_distances of all particles.
    """
    mean_distances: np.ndarray = np.full(shape=(swarm.n_particles),
                                         fill_value=0.0)
    for i, position in enumerate(swarm.position):
        # both have shape (n_particles - 1, dimesnions)
        curr_position: np.ndarray = np.full(
            shape=(swarm.n_particles - 1, swarm.dimensions),
            fill_value=position,
        )
        other_positions: np.ndarray = np.delete(swarm.position, i, axis=0)
        # distances have shape (n_particles - 1)
        distances: np.ndarray = np.linalg.norm(curr_position - other_positions,
                                               axis=1)
        mean_distance: float = np.mean(distances)
        mean_distances[i] = mean_distance
    return mean_distances
