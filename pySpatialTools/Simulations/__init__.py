
"""
Simulations
===========
Package for spatial simulations.


f function to update. Receive 2 vectors which descrive the spatial elements
which interacts and creates and ouput.
e spatial elements
Net connections between spatial elements


Required
--------
- SpatialModelEvolver
- State system (object which contains all the function needed to represent the
    system.)
- StateEvolve (object to compute the next state given a previous state)

"""

import numpy as np


class SpatialModelEvolver:
    """General class to spatial simulations.
    """

    def __init__(self):
        pass

    def evolve_i(self, state_i):
        pass


def general_iterator(state0, compute_connections, update_step, N_t):
    "Framework function to compute a simulation."
    state = state0  # elements + features
    for t in range(N_t):
        Net = compute_connections(state)
        elements, features = update_step(state, Net)
    return state
