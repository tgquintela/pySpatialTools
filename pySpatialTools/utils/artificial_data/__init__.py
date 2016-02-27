
"""
Artificial data
===============

"""

import numpy as np

from artificial_point_locations import random_space_points,\
    random_transformed_space_points
from artificial_spatial_relations import randint_sparse_matrix,\
    generate_randint_relations, generate_random_relations_cutoffs


def create_random_image(shape, n_modes=1):
    image = np.zeros((shape[0], shape[1], n_modes))
    for i in range(n_modes):
        aux = [np.random.randint(0, 256, shape[1]) for j in range(shape[0])]
        aux = np.vstack(aux).T
        image[:, :, 0] = aux
    return image
