"""
test parallel tools
-------------------
testing num
"""

from pySpatialTools.utils.util_external import distribute_tasks, reshape_limits


def test():
    distribute_tasks(100, 1000)
    lims = distribute_tasks(100, 10)
    reshape_limits(lims, [0, 20])
