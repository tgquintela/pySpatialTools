
"""
Implicit retrievers
-------------------
Implicit defined retrievers grouped in this module.

"""

import numpy as np
from itertools import product
from sklearn.neighbors import KDTree
from retrievers import Retriever
from aux_retriever import DummyRetriever


###############################################################################
############################ Space-Based Retrievers ###########################
###############################################################################
class SpaceRetriever(Retriever):
    """Retriever of elements considering its spacial information from a pool
    of elements retrievable.
    """
    typeret = 'space'

    def __init__(self, locs, info_ret=None, autolocs=None, pars_ret=None,
                 autoexclude=True, ifdistance=False, info_f=None,
                 perturbations=None, relative_pos=None, input_map=None,
                 output_map=None):
        "Creation a element space retriever class method."
        ## Reset globals
        self._initialization()
        # Output information
        self._format_output_information(autoexclude, ifdistance, relative_pos)
        ## Retrieve information
        self._define_retriever(locs, pars_ret)
        ## Info_ret mangement
        self._format_retriever_info(info_ret, info_f)
        # Location information
        self._format_locs(locs, autolocs)
        # Perturbations
        self._format_perturbation(perturbations)
        # IO mappers
        self._format_maps(input_map, output_map)

    ############################ Auxiliar functions ###########################
    ###########################################################################
    def _format_maps(self, input_map, output_map):
        if input_map is not None:
            self._input_map = input_map
        if output_map is not None:
            self._output_map = output_map

    def _format_output_exclude(self, i_locs, neighs, dists, output, k=0):
        "Format output with excluding."
        neighs, dists = self._output_map[output](self, i_locs, (neighs, dists))
        neighs, dists = self._exclude_auto(i_locs, neighs, dists, k)
        return neighs, dists

    def _format_output_noexclude(self, i_locs, neighs, dists, output, k=0):
        "Format output without excluding the same i."
        neighs, dists = self._output_map[output](self, i_locs, (neighs, dists))
        return neighs, dists

    def _format_locs(self, locs, autolocs):
        self.data = None if autolocs is None else autolocs
        self._autodata = True if self.data is None else False

    def _format_neigh_info(self, neighs_info):
        "TODO: Extension for other type of data as shapepy objects."
        neighs, dists = np.array(neighs_info[0]), np.array(neighs_info[1])
        n_dim = 1 if len(dists.shape) == 1 else dists.shape[1]
        dists = dists.reshape((len(dists), n_dim))
        return neighs, dists


################################ K Neighbours #################################
###############################################################################
class KRetriever(SpaceRetriever):
    "Class which contains a retriever of K neighbours."
    _default_ret_val = 1

    def _retrieve_neighs_spec(self, point_i, kneighs, ifdistance=False, kr=0):
        "Function to retrieve neighs in the specific way we want."
        point_i = self._get_loc_i(point_i)
        res = self.retriever[kr].query(point_i, int(kneighs), ifdistance)
        if ifdistance:
            res = res[1][0], res[0][0]
        else:
            res = res[0], [[] for i in range(len(res[0]))]

        ## Correct for another relative spatial measure (Save time indexing)
        if self.relative_pos is not None:
            loc_neighs = np.array(self.retriever[kr].data)[res[0], :]
            res = self._apply_relative_pos(res, point_i, loc_neighs)
        return res

    def _check_proper_retriever(self):
        "Check the correctness of the retriever for this class."
        try:
            for k in range(self.k_perturb):
                _, kr = self._map_perturb(k)
                loc = self.retriever[kr].data[0]
                self.retriever[kr].query(loc, 2)
            check = True
        except:
            check = False
        return check

    def _define_retriever(self, locs, pars_ret=None):
        "Define a kdtree for retrieving neighbours."
        if pars_ret is not None:
            leafsize = int(pars_ret)
        else:
            leafsize = locs.shape[0]
            leafsize = locs.shape[0]/100 if leafsize > 1000 else leafsize
        retriever = KDTree(locs, leaf_size=leafsize)
        self.retriever.append(retriever)


################################ R disctance ##################################
###############################################################################
class CircRetriever(SpaceRetriever):
    "Circular retriever."
    _default_ret_val = 0.1

    def _retrieve_neighs_spec(self, point_i, radius_i, ifdistance=False, kr=0):
        "Function to retrieve neighs in the specific way we want."
        point_i = self._get_loc_i(point_i)
        res = self.retriever[kr].query_radius(point_i, radius_i, ifdistance)
        if ifdistance:
            res = res[0][0], res[1][0]
        else:
            res = res[0], [[] for i in range(len(res[0]))]

        ## Correct for another relative spatial measure (Save time indexing)
        if self.relative_pos is not None:
            loc_neighs = np.array(self.retriever[kr].data)[res[0], :]
            res = self._apply_relative_pos(res, point_i, loc_neighs)
        return res

    def _check_proper_retriever(self):
        "Check the correctness of the retriever for this class."
        try:
            for k in range(self.k_perturb):
                _, kr = self._map_perturb(k)
                loc = self.retriever[kr].data[0]
                self.retriever[kr].query_radius(loc, 0.5)
            check = True
        except:
            check = False
        return check

    def _define_retriever(self, locs, pars_ret=None):
        "Define a kdtree for retrieving neighbours."
        if pars_ret is not None:
            leafsize = int(pars_ret)
        else:
            leafsize = locs.shape[0]
            leafsize = locs.shape[0]/100 if leafsize > 1000 else leafsize
        retriever = KDTree(locs, leaf_size=leafsize)
        self.retriever.append(retriever)


############################## Windows Neighbours #############################
###############################################################################
class WindowsRetriever(SpaceRetriever):
    """Class which contains a retriever of window neighbours for
    n-dimensional grid data.
    """
    _default_ret_val = {'l': 1, 'center': 0, 'excluded': False}

    def _retrieve_neighs_spec(self, element_i, pars_ret, ifdistance=False,
                              kr=0):
        """Retrieve all the neighs in the window described by pars_ret."""
        ## Get loc
        loc_i = self._get_loc_i(element_i, kr)
        neighs_info = generate_grid_neighs_coord(loc_i, self._shape,
                                                 self._ndim, **pars_ret)
        neighs = self.retriever[kr].map2indices(neighs_info[0])

        ## Compute neighs_info
        if ifdistance:
            neighs_info = neighs, neighs_info[1]
        else:
            neighs_info = neighs, None
        return neighs_info

    def _get_loc_i(self, element_i, kr):
        """Specific class function to get locations from input. Overwrite the
        generic function.
        """
        if type(element_i) == int:
            element_i = self.retriever[kr].map2locs(element_i)
        return element_i

    def _define_retriever(self, locs, pars_ret=None):
        "Define a kdtree for retrieving neighbours."
        # If it has to be excluded it will be excluded initially
        self._autoexclude = False
        self._format_output = self._format_output_noexclude
        ## If tuple of dimensions
        if type(locs) == tuple:
            types_ints = [type(locs[i]) == int for i in range(len(locs))]
            if np.all(types_ints):
                ndim = len(locs)
                limits = np.zeros((ndim, 2))
                for i in range(len(locs)):
                    limits[i] = np.array([0, locs[i]-1])
                shape_ = locs[:]
            else:
                raise TypeError("Incorrect locs input.")
        else:
            ndim = locs.shape[1]
            limits = np.zeros((ndim, 2))
            for i in range(ndim):
                limits[i] = np.array([locs[:, i].min(), locs[:, i].max()])
            shape_ = [int(limits[i][1]-limits[i][0]) for i in range(ndim)]
            shape_ = tuple(shape_)

        ## Create function for mapping
        shapes = np.array(list(np.cumprod(shape_[1:][::-1])[::-1]) + [1])

        def map2indices(x):
#            assert(len(x) == ndim)
            # Check if there is a correct coordinate
#            if np.all(x >= np.array(shape_)):
#                raise IndexError("Indices out of bounds.")
            try:
                idx = np.sum(x*shapes, 1).astype(int)
            except:
                idx = int(np.sum(x*shapes))
            return idx

        def map2locs(idx):
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
                n = np.prod(self.shape)
                ndim = len(self.shape)
                locs = np.zeros((n, ndim)).astype(int)
                dims = [xrange(self.shape[i]) for i in range(ndim)]
                for p in product(*dims):
                    i = self.map2indices(np.array(p))
                    locs[i] = np.array(p)
                return locs

            def __len__(self):
                return np.prod(self.shape)

        ## Store in a correct format
        self.retriever.append(WindRetriever(shape_, map2indices, map2locs))
        self._limits = limits
        self._shape = shape_
        self._ndim = len(limits)


def generate_grid_neighs_coord(coord, shape, ndim, l, center=0,
                               excluded=False):
    """Generation of neighbours from a point and the pars_ret.
    """
    if '__len__' in dir(l):
        window_l = np.array(l)
    else:
        window_l = np.array(ndim*[l])
    if '__len__' in dir(center):
        center = np.array(center)
    else:
        center = np.array(ndim*[center])
    try:
        center + window_l + coord
    except:
        raise TypeError("Incorrect parameters for window retriever.")
    ret_coord = coord + center
    if excluded:
        windows = []
        for i in range(ndim):
            ws = []
            for w in range(-int(window_l[i])/2, int(window_l[i])/2):
                x = coord[i] + center[i] + w
                if x >= 0 and x < shape[i] and (center[i]+w) != 0:
                    ws.append(x)
            windows.append(tuple(ws))
    else:
        windows = []
        for i in range(ndim):
            ws = []
            for w in range(-int(window_l[i])/2, int(window_l[i])/2):
                x = coord[i] + center[i] + w
                if x >= 0 and x < shape[i]:
                    ws.append(x)
            windows.append(tuple(ws))
    n_nei = np.prod([len(w) for w in windows])

    neighs_coord = np.zeros((n_nei, ndim)).astype(int)
    rel_pos = np.zeros((n_nei, ndim))
    i = 0
    for p in product(*windows):
        neighs_coord[i] = np.array(p)
        rel_pos = np.array(p) - ret_coord
#        rel_pos[i] = np.array(p)
#        neighs_coord[i] = ret_coord + rel_pos[i]
        i += 1
    return neighs_coord, rel_pos
