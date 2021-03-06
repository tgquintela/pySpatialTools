
"""
Artificial grid data
--------------------
Artificial data of grid data.

"""

import numpy as np


def create_random_image(shape, n_modes=1):
    """Create random image.

    Parameters
    ----------
    shape: tuple
        the size of the image.
    n_modes: int (default=1)
        the number of modes we want to generate the image.

    Returns
    -------
    image: np.ndarray, shape (n_in, n_out, n_modes)
        the random image with the shape given.

    """
    image = np.zeros((shape[0], shape[1], n_modes))
    for i in range(n_modes):
        aux = [np.random.randint(0, 256, shape[1]) for j in range(shape[0])]
        aux = np.vstack(aux).T
        image[:, :, 0] = aux
    return image
