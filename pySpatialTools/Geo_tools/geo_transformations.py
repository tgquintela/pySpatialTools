
"""
Module to transform geographical coordinates between magnitude transformations
or geographical projections.
"""

import numpy as np


def general_projection(data, loc_vars, method='ellipsoidal', inverse=False,
                       radians=False):
    "General global projection in order to compute distances."
    coordinates = data.loc[:, loc_vars].as_matrix()

    # Compute to correct magnitude (radians)
    if not radians:
        coordinates = degrees2radians(coordinates)
    ## Projection
    if method == 'spheroidal':
        coordinates = spheroidal_projection(coordinates, inverse)
    elif method == 'ellipsoidal':
        coordinates = ellipsoidal_projection(coordinates, inverse)
    # Correction magnitudes if inverse
    if inverse:
        coordinates = radians2degrees(coordinates)

    return coordinates


def radians2degrees(coordinates):
    "Transformation from radians to degrees."
    return 180./np.pi*coordinates


def degrees2radians(coordinates):
    "Transformation from degrees to radians."
    return np.pi/180.*coordinates


def spheroidal_projection(coordinates, inverse=False):
    "Projection under the assumption of spheroidal surface."
    coordinates[:, 0] = coordinates[:, 0]*np.cos(coordinates[:, 1])
    coordinates[:, 1] = coordinates[:, 1]
    return coordinates


def ellipsoidal_projection(coordinates, inverse=False):
    "Projection under the assumption of ellipsoidal surface."
    ## Constants measured experimentally
    K11, K12, K13 = 111.13209, -0.56605, 0.00120
    K21, K22, K23 = 111.41513, -0.09455, 0.00012

    aux0 = coordinates[:, 0]
    aux1 = coordinates[:, 1]
    ## Projection
    aux0 = (K21*np.cos(aux1)+K22*np.cos(3*aux1)+K23*np.cos(5*aux1))*aux0
    aux1 = (K11+K12*np.cos(2*aux1)+K13*np.cos(4*aux1))*aux1
    aux0 = 180./np.pi*aux0
    aux1 = 180./np.pi*aux1

    coordinates[:, 0] = aux0
    coordinates[:, 1] = aux1

    return coordinates
