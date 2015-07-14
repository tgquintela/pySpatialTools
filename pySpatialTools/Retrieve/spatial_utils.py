
"""
Spatial utils
=============
Module which groups the functions related with spatial utils which can be
useful.

"""

import numpy as np


def mapping2grid(locs, grid_size, xlim=(None, None), ylim=(None, None)):
    "Main function to map a group of points in a 2d to a grid."

    ## 1. Grid creation
    x, y = create_grid(locs, grid_size, xlim, ylim)
    xv, yv = np.meshgrid(x, y)

    ## 2. Application of the grid
    locs_agg_grid = apply_grid(locs, x, y)
    return locs_agg_grid, xv, yv


def create_grid(grid_size, locs=None, xlim=(None, None), ylim=(None, None)):
    ## 0. Preparation needed variables
    nx, ny = grid_size
    xmin, xmax = xlim
    ymin, ymax = ylim
    xmin = xmin if xmin is not None else locs[:, 0].min()
    xmax = xmax if xmax is not None else locs[:, 0].max()
    ymin = ymin if ymin is not None else locs[:, 1].min()
    ymax = ymax if ymax is not None else locs[:, 1].max()

    ## 1. Creation of the grid points
    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)
    return x, y


def apply_grid(locs, x, y):
    locs_agg_grid = -1*np.ones(locs.shape).astype(int)
    for i in xrange(locs.shape[0]):
        locs_agg_grid[i, 0] = mask_application(locs[i, 0], x)
        locs_agg_grid[i, 1] = mask_application(locs[i, 1], y)
    return locs_agg_grid


def mask_application(p, points):
    "Returns the index in which is placed the point."
    if p < points[0] or p > points[-1]:
        return -1
    for i in xrange(points.shape[0]-1):
        if p <= points[i+1]:
            return i
