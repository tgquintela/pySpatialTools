
"""
mappers vals i
--------------
Some examples of mappers for vals i
"""

from pySpatialTools.utils.util_classes import create_mapper_vals_i


def null_mapper_vals_i():
    mapper = create_mapper_vals_i(None, lambda s, i, k: i)
    return mapper


def corr_mapper_vals_i(corr_array):
    mapper = create_mapper_vals_i(corr_array, 'correlation')
    return mapper
