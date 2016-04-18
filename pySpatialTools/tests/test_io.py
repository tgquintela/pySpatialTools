
"""
Testing io
----------
Testing io utilities.
"""

import numpy as np
from pySpatialTools.io import create_locs_features_from_image


def test():
    sh = 50, 50, 3
    image = np.random.randint(0, 256, np.prod(sh)).reshape(sh)
    locs, feats = create_locs_features_from_image(image)
    sh = 50, 50
    image = np.random.randint(0, 256, np.prod(sh)).reshape(sh)
