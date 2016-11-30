
"""
io images
---------
Module to format images to be used by this framework.

"""

import numpy as np
from itertools import product


def create_locs_features_from_image(image):
    """Create locs and features from image.

    Parameters
    ----------
    image: np.ndarray
        the image matrix represented using numpy.

    Returns
    -------
    locs: np.ndarray
        the locations positions. The grid positions of the image.
    feats: np.ndarray
        the intensity of the image.

    """
    sh = image.shape
    if len(sh) == 2:
        image = image.reshape((sh[0], sh[1], 1))

    map2indices = lambda x, y: x + y*sh[0]

    n = np.prod(sh[:2])
    locs = np.zeros((n, 2)).astype(int)
    feats = np.zeros((n, image.shape[2])).astype(int)
    for p in product(xrange(sh[0]), xrange(sh[1])):
        i = map2indices(p[0], p[1])
        locs[i] = np.array([p[0], p[1]])
        feats[i] = image[p[0], p[1], :]
    return locs, feats
