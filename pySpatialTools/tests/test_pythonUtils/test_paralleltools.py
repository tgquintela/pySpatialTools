"""
test parallel tools
-------------------
testing num
"""

import numpy as np
from pySpatialTools.utils.util_external import distribute_tasks,\
    reshape_limits, generate_split_array, split_parallel


def test():
    distribute_tasks(100, 1000)
    lims = distribute_tasks(100, 10)
    reshape_limits(lims, [0, 20])
    generate_split_array([0, 20], lims)
    split_parallel(np.arange(1000), 100)
