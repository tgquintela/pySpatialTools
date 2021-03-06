
"""
Artificial data
===============
Module which groups artificial random data creation functions.

"""

import numpy as np

## Artificial random spatial locations
from artificial_point_locations import random_space_points,\
    random_transformed_space_points

## Artificial random spatial relations
from artificial_spatial_relations import randint_sparse_matrix,\
    generate_randint_relations, generate_random_relations_cutoffs
from artificial_data_membership import random_membership, list_membership

## Artificial random regions
from artificial_grid_data import create_random_image
from artificial_regions_data import random_shapely_polygon,\
    random_shapely_polygons

## Artificial random features
from artificial_features import continuous_array_features,\
    categorical_array_features, continuous_dict_features,\
    categorical_dict_features, continuous_agg_array_features,\
    categorical_agg_array_features, continuous_agg_dict_features,\
    categorical_agg_dict_features

## Artificial random measures
from artificial_measure import create_vals_i, create_artificial_measure_array,\
    create_artificial_measure_replacelist, create_artificial_measure_append,\
    create_empty_array, create_empty_append, create_empty_replacelist,\
    create_empty_features_array, create_empty_features_dict,\
    create_features_i_array, create_features_i_dict
