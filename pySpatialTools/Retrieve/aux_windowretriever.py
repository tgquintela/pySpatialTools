
"""
auxiliar grid retriever
-----------------------
Module which groups all the auxiliar external functions for window retrieving
in a regular spatial data.
It helps in retrieving by groups of elements with the same number of neighs.

"""

import numpy as np
from itertools import product, combinations
from pySpatialTools.utils.util_external.parallel_tools import split_parallel

## TO SOLVE:
## Negative numbers
## Split parallel


########################### Main auxiliar functions ###########################
###############################################################################
def create_window_utils(shape_):
    """Create the basic utilities for windows retriever.

    Parameters
    ----------
    shape_: tuple
        the size of the grid.

    Returns
    -------
    map2indices: function
        function which maps the spatial information to indices.
    map2locs: function
        function which maps the indices to spatial information.
    WindRetriever: instance Object
        the core-retriever for windows retrieving.

    """
    ## Create function for mapping
    shapes = np.array(list(np.cumprod(shape_[1:][::-1])[::-1]) + [1])

    def map2indices(x):
        """Mapper function which transforms locations into indices.

        Parameters
        ----------
        x: np.ndarray
            the locations.

        Returns
        -------
        idx: indices
            the indices related with x locations.

        """
#        assert(len(x) == ndim)
#        # Check if there is a correct coordinate
#        if np.all(x >= np.array(shape_)):
#            raise IndexError("Indices out of bounds.")
        try:
            idx = np.sum(x*shapes, 1).astype(int)
        except:
            idx = int(np.sum(x*shapes))
        return idx

    def map2locs(idx):
        """Mapper function which transforms indices into locations.

        Parameters
        ----------
        idx: indices
            the indices related with coord locations.

        Returns
        -------
        coord: np.ndarray
            the locations related with the given indices.

        """
        if idx < 0 or idx >= np.prod(shape_):
            raise IndexError("Indices out of bounds.")
        coord = np.zeros(len(shapes))
        for j in range(len(shapes)):
            coord[j] = idx/shapes[j]
            idx = idx % shapes[j]
        assert(idx == 0)
        return coord

    ## Create class
    class WindRetriever:
        """Windows Object retriever."""
        def __init__(self, shape, map2indices, map2locs):
            self.map2indices = map2indices
            self.shape = shape
            self.map2locs = map2locs

        @property
        def data(self):
            """Retrivable data."""
            n = np.prod(self.shape)
            ndim = len(self.shape)
            locs = np.zeros((n, ndim)).astype(int)
            dims = [xrange(self.shape[i]) for i in range(ndim)]
            for p in product(*dims):
                i = self.map2indices(np.array(p))
                locs[i] = np.array(p)
            return locs

        def get_indices(self, loc):
            """Mapper function which transforms locations into indices.

            Parameters
            ----------
            x: np.ndarray
                the locations.

            Returns
            -------
            idx: indices
                the indices related with x locations.

            """
            return self.map2indices_iss(loc)

        def get_locations(self, idx):
            """Get locations from indices.

            Parameters
            ----------
            idx: indices
                the indices related with coord locations.

            Returns
            -------
            coord: np.ndarray
                the locations related with the given indices.

            """
            return self.map2locs_iss(idx)

        def __len__(self):
            """Number of retrievable elements."""
            return np.prod(self.shape)

        def map2indices_iss(self, xss):
            idxs_iss = []
            for i in range(len(xss)):
                idxs_iss.append(self.map2indices(xss[i]))
            return idxs_iss

        def map2locs_iss(self, iss):
            iss = [iss] if type(iss) == int else iss
            locs_iss = []
            for i in range(len(iss)):
                locs_iss.append(self.map2locs(iss[i]))
            locs_iss = np.array(locs_iss)
            return locs_iss

    return map2indices, map2locs, WindRetriever


def windows_iteration(shape, max_bunch, l, center, excluded):
    """Main iteration over all the grid by number of neighs.

    Parameters
    ----------
    shape: tuple
        the size of the grid.
    max_bunch: int
        maximum number of elements we want to retrieve their neighbours
        at the same time.
    l: int
        size of the windows to retrieve.
    center: int
        the place of the center.
    excluded: boolean (default = False)
        if we want to exclude the element `i`.

    Returns
    -------
    inds: np.ndarray
        the indices of the elements which we want to retrieve their neighbours.
    neighs: np.ndarray
        the neighbours of each inds.
    rel_pos: np.ndarray
        the relative positions of each neighbours to its retrieved element.

    """
    ## Splitting by special corner and border points
    shapes = np.array(list(np.cumprod(shape[1:][::-1])[::-1]) + [1])
    coord2ind = create_map2indices(shape)
    borders, ranges, sizes_nei =\
        get_irregular_indices_grid(shape, l, center, excluded)
    ## Retrieving by sizes of neighbourhood
    sizes_u = np.unique(sizes_nei)
    if len(sizes_u) != 1:
        for s in sizes_u:
            inds, neighs = [], []
            size_idx = np.where(s == sizes_nei)[0]
            for idx_s in size_idx:
                # Get indices of the special points
                inds_s = get_indices_from_borders(borders[idx_s], coord2ind)
                # Get relative position
                rel_pos = get_relative_neighs_comb(ranges[idx_s])
                # Get neigbours
                neighs_s = get_irregular_neighsmatrix(inds_s, rel_pos, shapes)
                inds.append(inds_s)
                neighs.append(neighs_s)
            ## Formatting properly
            inds, neighs = np.concatenate(inds), np.concatenate(neighs, axis=1)
            inds, neighs = inds.astype(int), neighs.astype(int)
            yield inds, neighs, rel_pos

    indices, relative_neighs, rel_pos =\
        get_indices_constant_regular(shape, coord2ind, l, center, excluded)
    indices_split = split_parallel(indices, max_bunch)
#    indices_split = [indices]
    for inds in indices_split:
        neighs = get_regular_neighsmatrix(inds, relative_neighs)
        yield inds, neighs, rel_pos


############################## Mapper coordinates #############################
###############################################################################
def create_map2indices(shape_):
    """Create the basic utilities for windows retriever.

    Parameters
    ----------
    shape_: tuple
        the size of the grid.

    Returns
    -------
    map2indices: function
        function which maps the spatial information to indices.

    """
    shapes = np.array(list(np.cumprod(shape_[1:][::-1])[::-1]) + [1])

    def map2indices(x):
        """Mapper function which transforms locations into indices.

        Parameters
        ----------
        x: np.ndarray
            the locations.

        Returns
        -------
        idx: indices
            the indices related with x locations.

        """
        # Check if there is a correct coordinate
        try:
            idx = np.sum(x*shapes, 1).astype(int)
        except:
            idx = int(np.sum(x*shapes))
        return idx
    return map2indices


################################ Main functions ###############################
###############################################################################
# Main function regular core
def get_indices_constant_regular(shape_, coord2ind, l, center, excluded=False):
    """Get the regular neighs matrix.

    Parameters
    ----------
    shape_: tuple
        the size of each dimension.
    coord2ind: function
        function which maps the spatial information to indices.
    l: int
        size of the windows to retrieve.
    center: int
        the place of the center.
    excluded: boolean (default = False)
        if we want to exclude the element `i`.

    Returns
    -------
    neighs_coord: np.ndarray
        the neighbours coordinates.
    rel_pos: np.ndarray
        the distance of each neighbours to its retrieved element.

    """
    ## Get extremes regular neighs
    c = center
    l = np.array(l) if '__len__' in dir(l) else np.array(len(shape_)*[l])
    c = np.array(c) if '__len__' in dir(c) else np.array(len(shape_)*[c])
    extremes = get_extremes_regularneighs_grid(shape_, l, c)
    indices = get_core_indices(extremes, coord2ind)
    relative_neighs = get_relative_neighs(shape_, l, c, excluded)
    rel_pos = get_relative_neighs_comb(relative_neighs)
    relative_neighs = [coord2ind(np.array(rel)) for rel in rel_pos]
#    indices = []
#    for dim in range(len(extremes_ind)):
#        indices.extend(range(extremes_ind[dim, 0], extremes_ind[dim, 1]+1))
#    indices = np.array(indices).reshape((len(indices), 1))
    return indices, relative_neighs, rel_pos


# Main function borders and corners
def get_irregular_indices_grid(shape_, l, center, excluded):
    """Get irregular information to get indices of the grid.

    Parameters
    ----------
    shape_: tuple
        the size of each dimension.
    l: int
        size of the windows to retrieve.
    center: int (default = 0)
        the place of the center.
    excluded: boolean (default = False)
        if we want to exclude the element `i`.

    Returns
    -------
    extremes: np.ndarray (dim, 2)
        the extremes for each dimension.
    ranges: list
        the consecutive elements for each dimension.
    sizes: np.ndarray
        the size of each bunch of elements.

    """
    c = center
    l = np.array(l) if '__len__' in dir(l) else np.array(len(shape_)*[l])
    c = np.array(c) if '__len__' in dir(c) else np.array(len(shape_)*[c])
    diff = get_relative_neighs(shape_, l, c, excluded)
    extremes, ranges = new_get_irregular_extremes(diff, shape_)
    extremes, ranges =\
        new_get_borders_from_irregular_extremes(extremes, shape_, diff)
    sizes = np.array([np.prod([len(e) for e in r]) for r in ranges])
    return extremes, ranges, sizes


############################### Get neighsmatrix ##############################
###############################################################################
def get_irregular_neighsmatrix(indices, relative_neighs, shapes):
    neighs = np.zeros((len(relative_neighs), len(indices))).astype(int)
    for i in range(len(relative_neighs)):
        rel_ind = np.sum(np.array(relative_neighs[i])*shapes)
        neighs[i] = indices + rel_ind
    return neighs


def get_regular_neighsmatrix(indices, relative_neighs):
    neighs = np.repeat(relative_neighs, len(indices))
    neighs = (neighs.reshape(len(relative_neighs), len(indices)) + indices).T
    return neighs


###################### Main auxiliar corners functions ########################
###############################################################################
def get_indices_from_borders(borders, coord2ind):
    """Get all the indices from the borders."""
    logi = np.array([isinstance(e, slice) for e in borders])
    ndim = len(borders)
    if np.any(logi):
        constants = np.array([borders[i] for i in range(ndim) if not logi[i]])
        slices = [xrange(borders[i].start, borders[i].stop)
                  for i in range(ndim) if logi[i]]
        coord = np.zeros(ndim).astype(int)
        indices = []
        coord[np.logical_not(logi)] = constants
        for p in product(*slices):
            for i in range(len(p)):
                coord[logi] = np.array(p)
                indices.append(coord2ind(coord))
        indices = np.array(indices).astype(int)

    else:
        indices = np.array([coord2ind(borders)])
    indices = np.array(indices).astype(int)
    return indices


def new_get_borders_from_irregular_extremes(extremes, shape, grange):
    """Get all the corners and border slices.

    Parameters
    ----------
    extremes: np.ndarray (dim, 2)
        the extremes for each dimension.
    shape: tuple
        the size of each dimension.
    grange: array_like (dim, elements)
        the information to generate ranges for each dimension.

    Returns
    -------
    points_corners: list
        the coordinates of the corners.
    ranges: list
        the consecutive elements for each dimension.

    """
    ndim = len(extremes)
    corners, ranges = [], []
    for dim in range(ndim):
        corners_dim = range(0, extremes[dim][0]+1)
        corners_dim += range(extremes[dim][-1], shape[dim])

        corners.append(list(set(corners_dim)))
    points_corners = []
    for p in product(*corners):
        points_corners.append(tuple(p))
        ranges_p = []
        for dim in range(ndim):
            ranges_dim = [e for e in grange[dim] if e >= -p[dim]
                          and e < shape[dim]-p[dim]]
            ranges_p.append(ranges_dim)
        ranges.append(ranges_p)

    for n_i in range(1, ndim):
        ## Possible combinations
        for dims in combinations(range(ndim), n_i):
            if not dims:
                continue
            dims = list(dims)
            no_dims = [i for i in range(ndim) if i not in dims]
            corners_nodims = [corners[nodim] for nodim in no_dims]
            slices = [slice(extremes[dim][0]+1, extremes[dim][1])
                      for dim in dims]
            for p in product(*corners_nodims):
                aux_bor = [[]]*ndim
                ranges_p = []
                for i in range(ndim):
                    if i in dims:
                        aux_bor[i] = slices[dims.index(i)]
                        ranges_p.append(grange[i])
                    else:
                        aux_bor[i] = p[no_dims.index(i)]
                        ranges_p.append([e for e in grange[i]
                                         if e >= -p[no_dims.index(i)]
                                         and e < shape[i]-p[no_dims.index(i)]])

                ranges.append(ranges_p)
                points_corners.append(tuple(aux_bor))
    return points_corners, ranges


def new_get_irregular_extremes(diff, shape):
    init_points, init_ranges = [], []
    for dim in range(len(diff)):
        init = -1 - diff[dim][0]
        init = 0 if init < 0 else init
        endit = shape[dim] - diff[dim][-1]
        endit = 0 if endit >= shape[dim] else endit
        init_points.append([init, endit])
        init_r = [dr for dr in diff[dim] if dr > diff[dim][0]]
        endit_r = [dr for dr in diff[dim] if dr < diff[dim][-1]]
        init_ranges.append([init_r, endit_r])
    points, ranges = init_points, init_ranges
    return points, ranges


#################### Main auxiliar central core functions #####################
###############################################################################
def get_core_indices(extremes, coord2ind):
    """Get the indices of the extremes.

    Parameters
    ----------
    extremes: np.ndarray (dim, 2)
        the extremes for each dimension.
    coord2ind: function
        function which maps the spatial information to indices.

    Returns
    -------
    indices: np.ndarray
        the indices of all the elements between that extremes.

    """
    ## 0. Get formatted indices of extrems
    ndim = len(extremes)
    core = [range(extremes[dim][0], extremes[dim][1]) for dim in range(ndim)]
    indices = np.zeros(np.prod([len(e) for e in core])).astype(int)
    i = 0
    for p in product(*core):
        indices[i] = coord2ind(np.array(p))
        i += 1
    return indices.ravel()


def get_extremes_regularneighs_grid(shape, l, center, excluded=False):
    """Get the extremes points of the regularneighs grid.

    Parameters
    ----------
    shape: tuple
        the size of each dimension.
    l: int
        size of the windows to retrieve.
    center: int (default = 0)
        the place of the center.
    excluded: boolean (default = False)
        if we want to exclude the element `i`.

    Returns
    -------
    extremes: np.ndarray (dim, 2)
        the extremes for each dimension.

    """
    w_l, c = l, center
    w_l = np.array(l) if '__len__' in dir(l) else np.array(len(shape)*[l])
    c = np.array(c) if '__len__' in dir(c) else np.array(len(shape)*[c])

    extremes = np.zeros((len(shape), 2)).astype(int)
    for dim in range(len(shape)):
        extremes[dim, 0] = w_l[dim]/2-c[dim] if w_l[dim]/2-c[dim] > 0 else 0
        endit = shape[dim] - (w_l[dim]/2+c[dim])
        extremes[dim, 1] = endit if endit < shape[dim] else shape[dim]-1
    return extremes


######################### General auxiliar functions ##########################
###############################################################################
def get_relative_neighs(shape, l, c, excluded):
#    shapes = np.array(list(np.cumprod(shape[1:][::-1])[::-1]) + [1])
    diff = []
    for dim in range(len(shape)):
        a = [i for i in range(-l[dim]/2+c[dim]+1, l[dim]/2+c[dim]+1)
             if i != 0 or not excluded]
        diff.append(a)
#    relative_neighs = []
#    for p in product(*diff):
#        relative_neighs.append(int(np.sum(np.array(p)*shapes)))
    return diff


def get_relative_neighs_comb(ranges):
    relative_pos = []
    for p in product(*[range(len(e)) for e in ranges]):
        relative_pos.append([ranges[i][p[i]] for i in range(len(p))])
    return relative_pos


def generate_grid_neighs_coord(coords, shape, ndim, l, center=0,
                               excluded=False):
    """Generation of neighs for different coords.

    Parameters
    ----------
    coords: np.ndarray
        the coordinates we want to obtain its neighs.
    shape: tuple
        the size of each dimension.
    ndim: int
        the number of dimensions.
    l: int
        size of the windows to retrieve.
    center: int (default = 0)
        the place of the center.
    excluded: boolean (default = False)
        if we want to exclude the element `i`.

    Returns
    -------
    neighs_coords: np.ndarray
        the neighbours coordinates.
    rel_poss: np.ndarray
        the distance of each neighbours to its retrieved element.

    """
    ## Preparation inputs
    coords = np.array(coords).astype(int)
    if len(coords.shape) != 2:
        coords = coords.reshape((1, ndim))
    ## Computing
    neighs_coords, rel_poss = [], []
    for i in range(len(coords)):
        neighs_coord, rel_pos =\
            generate_grid_neighs_coord_i(coords[i], shape, ndim, l, center,
                                         excluded)
        neighs_coords.append(neighs_coord)
        rel_poss.append(rel_pos)
    return neighs_coords, rel_poss


def generate_grid_neighs_coord_i(coord, shape, ndim, l, center=0,
                                 excluded=False):
    """Generation of neighbours from a point and the pars_ret.

    Parameters
    ----------
    coord: np.ndarray
        the coordinates we want to obtain its neighs.
    shape: tuple
        the size of each dimension.
    ndim: int
        the number of dimensions.
    l: int
        size of the windows to retrieve.
    center: int (default = 0)
        the place of the center.
    excluded: boolean (default = False)
        if we want to exclude the element `i`.

    Returns
    -------
    neighs_coord: np.ndarray
        the neighbours coordinates.
    rel_pos: np.ndarray
        the relative positions of each neighbours to its retrieved element.

    """
    ## Format and check inputs
    coord = coord.ravel()
    c = center
    window_l = np.array(l) if '__len__' in dir(l) else np.array(ndim*[l])
    center = np.array(c) if '__len__' in dir(c) else np.array(ndim*[c])
    try:
        center + window_l + coord
    except:
        raise TypeError("Incorrect parameters for window retriever.")
    ## Compute
    ret_coord = coord + center
    if excluded:
        windows = []
        for i in range(ndim):
            ws = []
            for w in range(-(int(window_l[i])/2), int(window_l[i]+1)/2):
                x = coord[i] + center[i] + w
                if x >= 0 and x < shape[i] and (center[i]+w) != 0:
                    ws.append(x)
            windows.append(tuple(ws))
    else:
        windows = []
        for i in range(ndim):
            ws = []
            for w in range(-(int(window_l[i])/2), int(window_l[i]+1)/2):
                x = coord[i] + center[i] + w
                if x >= 0 and x < shape[i]:
                    ws.append(x)
            windows.append(tuple(ws))
    n_nei = np.prod([len(w) for w in windows])

    neighs_coord = np.zeros((n_nei, ndim)).astype(int)
    rel_pos = np.zeros((n_nei, ndim)).astype(int)
    i = 0
    for p in product(*windows):
        neighs_coord[i] = np.array(p)
        rel_pos[i] = np.array(p) - ret_coord
#        rel_pos[i] = np.array(p)
#        neighs_coord[i] = ret_coord + rel_pos[i]
        i += 1
    return neighs_coord, rel_pos
