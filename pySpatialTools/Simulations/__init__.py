
"""
Simulations
===========
Package for spatial simulations.


f function to update. Receive 2 vectors which descrive the spatial elements
which interacts and creates and ouput.
e spatial elements
Net connections between spatial elements

"""

import numpy as np


def general_iterator(state0, compute_connections, update_step, N_t):
    "Framework function to compute a simulation."
    state = state0  # elements + features
    for t in range(N_t):
        Net = compute_connections(state)
        elements, features = update_step(state, Net)
    return state
