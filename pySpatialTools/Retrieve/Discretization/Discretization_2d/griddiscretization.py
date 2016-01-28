
"""
Grid discretizer
----------------
The class and functions related with the 2d-grid discretization.

"""

import numpy as np
from scipy.sparse import coo_matrix

from pySpatialTools.Retrieve.Discretization.spatialdiscretizer import \
    SpatialDiscretizor
from pySpatialTools.Retrieve.Discretization.spatial_utils import \
    mask_application


################################# Grid based ##################################
###############################################################################
class GridSpatialDisc(SpatialDiscretizor):
    "Grid spatial discretization. The regions are rectangular with equal size."

    def __init__(self, grid_size, xlim=(None, None), ylim=(None, None)):
        "Main function to map a group of points in a 2d to a grid."
        self.create_grid(grid_size, xlim=xlim, ylim=ylim)
        self.compute_limits()

    ########################### Automatic Functions ##########################
    ##########################################################################
    def create_grid(self, grid_size, xlim=(None, None), ylim=(None, None)):
        "Create a grid with the parameters we want."
        self.borders = create_grid(grid_size=grid_size, xlim=xlim, ylim=ylim)

    ## Automatic
    def compute_limits(self, region_id=None):
        "Build the limits of the region discretized."
        if region_id is None:
            limits = compute_limits_grid(self.borders)
            self.limits = limits
        else:
            limits = compute_limits_grid(self.borders, region_id)
            return limits

    ################################ Functions ###############################
    ##########################################################################
    def get_nregs(self):
        "Get number of regions."
        n_regs = (self.borders[0].shape[0]-1, self.borders[1].shape[0]-1)
        n_regs = np.prod(n_regs)
        return n_regs

    def get_regions_id(self):
        "Get regions id described by this discretizor."
        return np.arange(self.get_nregs())

    def get_regionslocs(self):
        "Get the regionslocs (representative region location) described here."
        return self.map_regionid2regionlocs(self.get_regions_id())

    ########################### Compulsary mappers ###########################
    ##########################################################################
    def map_loc2regionid(self, locs):
        """Discretize locs returning their region_id.

        Parameters
        ----------
        locs: numpy.ndarray
            the locations for which we want to obtain their region given that
            discretization.

        Returns
        -------
        regions: numpy.ndarray
            the region_id of each locs for this discretization.

        """
        ## Discretize locations
        locs = locs.reshape(1, locs.shape[0]) if len(locs.shape) == 1 else locs
        locs_grid = apply_grid(locs, self.borders[0], self.borders[1])
        ## Mapping to region ID
        grid_size = (self.borders[0].shape[0]-1, self.borders[0].shape[0]-1)
        regions_id = map_gridloc2regionid(locs_grid, grid_size)
        return regions_id

    def map_locs2regionlocs(self, locs):
        "Map locations to regionlocs."
        grid_locs = self.apply_grid(locs)
        regionlocs = map_gridloc2regionlocs(grid_locs, self.borders)
        return regionlocs

    def map_regionid2regionlocs(self, regions):
        """Function which maps the regions ID to their most representative
        location.
        """
        ## 0. Needed variables
        regions = np.array([regions]) if type(regions) == int else regions
        grid_size = (self.borders[0].shape[0]-1, self.borders[0].shape[0]-1)
        ## 1. Discretization of grid locations
        grid_locs = map_regionid2gridloc(regions, grid_size)
        ## 2. Agglocations
        regionlocs = map_gridloc2regionlocs(grid_locs, self.borders)
        return regionlocs

    ########################## Contiguity definition #########################
    ##########################################################################
    def compute_contiguity_geom(self, region_id=None):
        "Compute which regions are contiguous and returns a graph."
        ## 0. Compute needed variables
        nx, ny = self.borders[0].shape[0]-1, self.borders[1].shape[0]-1
        ## 1. Only compute neighs of a specific region if it is required
        if region_id is not None:
            return compute_contiguity_grid(region_id, (nx, ny))
        ## 2. Compute contiguity for all regions
        iss, jss, dts = [], [], []
        for i in range(nx):
            for j in range(ny):
                aux_ = compute_contiguity_grid(i*nx+j, (nx, ny))
                n_aux = len(aux_)
                dts.append(np.ones(n_aux).astype(int))
                iss.append(np.ones(n_aux).astype(int)*i*nx+j)
                jss.append(np.array(aux_).astype(int))
        iss, jss = np.hstack(iss).astype(int), np.hstack(jss).astype(int)
        dts = np.hstack(dts).astype(int)
        contiguous = coo_matrix((dts, (iss, jss)), shape=(nx*ny, nx*ny))
        return contiguous

    ##################### Definition of particularities ######################
    ##########################################################################
    def apply_grid(self, locs):
        """Discretize locs given their region_id.

        Parameters
        ----------
        locs: numpy.ndarray
            the locations for which we want to obtain their region given that
            discretization.

        Returns
        -------
        locs_grid: numpy.ndarray
            the region discretized coordinates.

        """
        locs_grid = apply_grid(locs, self.borders[0], self.borders[1])
        return locs_grid


###############################################################################
############################# Auxiliary functions #############################
###############################################################################
def compute_limits_grid(borders, region_id=None):
    "Compute the limits of the region or the whole space discretized."
    limits = np.zeros((2, 2))
    if region_id is None:
        limits[0, 0] = borders[0].min()
        limits[0, 1] = borders[0].max()
        limits[1, 0] = borders[1].min()
        limits[1, 1] = borders[1].max()
    else:
        grid_size = borders[0].shape[0]-1, borders[1].shape[0]-1
        grid_loc = map_regionid2gridloc(region_id, grid_size)
        limits[0, 0] = borders[0][grid_loc[0, 0]]
        limits[0, 1] = borders[0][grid_loc[0, 0]+1]
        limits[1, 0] = borders[1][grid_loc[0, 1]]
        limits[1, 1] = borders[1][grid_loc[0, 1]+1]
    return limits


def compute_contiguity_grid(region_id, grid_size):
    "Compute the contiguity of the regions."
    nx, ny = grid_size
    contiguous = []
    if region_id >= nx*ny:
        return []
    if (region_id - nx) >= 0:
        contiguous.append(region_id - nx)
    if (region_id + nx) < nx*ny:
        contiguous.append(region_id + nx)
    if ((region_id - 1) % nx) < (region_id % nx):
        contiguous.append(region_id - 1)
    if ((region_id + 1) % nx) > (region_id % nx):
        contiguous.append(region_id + 1)
    return contiguous


def map_gridloc2regionid(locs_grid, grid_size):
    "Transformation: grid_loc --> region_id."
    return locs_grid[:, 0]*grid_size[0]+locs_grid[:, 1]


def map_regionid2gridloc(regions_id, grid_size):
    "Transformation: region_id --> grid_loc."
    n = grid_size[0]
    return np.vstack([regions_id / n, regions_id % n]).T


def map_gridloc2regionlocs(grid_locs, borders):
    "Transformation: grid_loc --> regionlocs."
    regionlocs = np.zeros(grid_locs.shape).astype(float)
    delta_x = borders[0][1]-borders[0][0]
    delta_y = borders[1][1]-borders[1][0]
    regionlocs[:, 0] = grid_locs[:, 0]*delta_x + delta_x/2.
    regionlocs[:, 1] = grid_locs[:, 1]*delta_y + delta_y/2.
    return regionlocs


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
    nx, ny = nx+1, ny+1
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
