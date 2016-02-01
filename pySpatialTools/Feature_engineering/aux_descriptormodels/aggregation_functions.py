
"""
aggregation functions
---------------------
This module contains aggregation functions to be used by the aggregation
function of the features object.

"""

#import numpy as np
from characterizers import characterizer_1sh_counter, characterizer_summer,\
    characterizer_average


def aggregator_1sh_counter(pointfeats, point_pos):
    """Aggregator which counts the different types of elements in the
    aggregation units."""
    descriptors = characterizer_1sh_counter(pointfeats, point_pos)
    return descriptors


def aggregator_summer(pointfeats, point_pos):
    """Aggregator which sums the different element features in the
    aggregation units."""
    descriptors = characterizer_summer(pointfeats, point_pos)
    return descriptors


def aggregator_average(pointfeats, point_pos):
    """Aggregator which average the different element features in the
    aggregation units."""
    descriptors = characterizer_average(pointfeats, point_pos)
    return descriptors
