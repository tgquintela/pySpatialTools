
"""
Locations preprocess
--------------------
Module which groups all the methods to preprocess locations.
"""

import numpy as np


def remove_unknown_locations(locations, logi):
    """Remove unknown locations."""
    return locations[logi]


def jitter_group_imputation(locations, logi, groups):
    """Jitter the average locations of the group.

    Parameters
    ----------
    locations: np.ndarray, shape (n, 2)
        the locations information.
    logi: boolean np.ndarray, shape (n)
        the locations which are uncorrect.
    groups: integer np.ndarray, shape (n)
        the groups in which we want to obtain their own standard deviation
        in order to create a random loation.

    Returns
    -------
    new_locations: np.ndarray
        the new locations.

    """
    if np.all(logi):
        return locations
    u_groups, new_locations = np.unique(groups), np.zeros(locations.shape)
    new_locations[logi] = locations[logi]
    for g in u_groups:
        ## Select group
        logi_unk_g = np.logical_and(np.logical_not(logi), groups == g)
        locs_g = locations[np.logical_and(logi, groups == g)]
        ## Statistics of the group
        loc_g, std_g = np.mean(locs_g, axis=0), np.std(locs_g, axis=0)
        ## Build new coordinates
        jitter_g = np.random.random((np.sum(logi_unk_g), locations.shape[1]))
        new_coordinates = np.multiply(std_g, jitter_g) + loc_g
        ## Imputation
        new_locations[logi_unk_g] = new_coordinates
    return new_locations
