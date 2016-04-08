
"""
Implicit retrievers
-------------------
Implicit defined retrievers grouped in this module.

"""

import numpy as np
#from itertools import product
from sklearn.neighbors import KDTree
from retrievers import Retriever
from aux_retriever import DummyRetriever
from aux_windowretriever import generate_grid_neighs_coord,\
    create_window_utils, windows_iteration
from pySpatialTools.utils.util_external.parallel_tools import split_parallel


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
                 output_map=None, constant_info=False, bool_input_idx=None,
                 format_level=None, type_neighs=None, type_sp_rel_pos=None):
        "Creation a element space retriever class method."
        ## Reset globals
        self._initialization()
        # Output information
        self._format_output_information(autoexclude, ifdistance, relative_pos)
        self._format_exclude(bool_input_idx)
        ## Retrieve information
        self._define_retriever(locs, pars_ret)
        ## Info_ret mangement
        self._format_retriever_info(info_ret, info_f, constant_info)
        ## Format retriever function
        self._format_retriever_function(bool_input_idx)
        # Location information
        self._format_locs(locs, autolocs)
        # Perturbations
        self._format_perturbation(perturbations)
        # IO mappers
        self._format_maps(input_map, output_map)
        self._format_preparators(bool_input_idx)
        self._format_neighs_info(bool_input_idx, format_level, type_neighs,
                                 type_sp_rel_pos)

    ############################ Auxiliar functions ###########################
    ###########################################################################
    def _format_maps(self, input_map, output_map):
        if input_map is not None:
            self._input_map = input_map
        if output_map is not None:
            self._output_map = output_map

    def _format_output_exclude(self, i_locs, neighs, dists, output=0, k=0):
        "Format output with excluding."
        neighs, dists = self._output_map[output](self, i_locs, (neighs, dists))
        neighs, dists = self._exclude_auto(i_locs, neighs, dists, k)
        return neighs, dists

    def _format_output_noexclude(self, i_locs, neighs, dists, output=0, k=0):
        "Format output without excluding the same i."
        neighs, dists = self._output_map[output](self, i_locs, (neighs, dists))
        return neighs, dists

    def _format_locs(self, locs, autolocs):
        self.data = None if autolocs is None else autolocs
        self._autodata = True if self.data is None else False

#    def _format_neigh_info(self, neighs_info):
#        "TODO: Extension for other type of data as shapepy objects."
#        pass
#        neighs, dists = np.array(neighs_info[0]), np.array(neighs_info[1])
#        n_dim = 1 if len(dists.shape) == 1 else dists.shape[1]
#        dists = dists.reshape((len(dists), n_dim))
#        return neighs, dists


################################ K Neighbours #################################
###############################################################################
class KRetriever(SpaceRetriever):
    "Class which contains a retriever of K neighbours."
    _default_ret_val = 1
    constant_neighs = True
    preferable_input_idx = False
    auto_excluded = True
    output_array = True

    ######################## Retrieve-driven retrieve #########################
    def set_iter(self, info_ret=None, max_bunch=None):
        info_ret = self._default_ret_val if info_ret is None else info_ret
        max_bunch = len(self) if max_bunch is None else max_bunch
        self._info_ret = info_ret
        self._max_bunch = max_bunch

    def __iter__(self):
        ## Prepare iteration
        bool_input_idx = True
        self._format_preparators(bool_input_idx)
        self._constant_ret = True
        self._format_retriever_function(bool_input_idx)
        ## Prepare indices
        indices = split_parallel(np.arange(self._n0), self._max_bunch)
        ## Iteration
        for idxs in indices:
            neighs = self.retrieve_neighs(idxs)
            yield indices, neighs

    ###################### Retrieve functions candidates ######################
    def _retrieve_neighs_general_spec(self, point_i, kneighs, ifdistance=False,
                                      kr=0):
        """General function to retrieve neighs in the specific way we want."""
        point_i = self._prepare_input(point_i, kr)
        res = self.retriever[kr].query(point_i, int(kneighs), ifdistance)
        if ifdistance:
            res = np.array(res[1]), list(res[0])
            res = res[0], self._apply_preprocess_relative_pos(res[1])
        else:
            res = np.array(res), None
        ## Correct for another relative spatial measure (Save time indexing)
        res = self._apply_relative_pos_spec(res, point_i)
        return res

    def _retrieve_neighs_constant_nodistance(self, point_i, info_i={}, kr=0):
        """Retrieve neighs not computing distance by default.

        Parameters
        ----------
        point_i: int
            the indice of the point_i.
        """
        kneighs = self._get_info_i(point_i, info_i)
        point_i = self._prepare_input(point_i, kr)
        res = self.retriever[kr].query(point_i, int(kneighs), False)
        res = np.array(res), None
        return res

    def _retrieve_neighs_constant_distance(self, point_i, info_i={}, kr=0):
        """Retrieve neighs not computing distance by default.

        Parameters
        ----------
        point_i: int
            the indice of the point_i.
        """
        if type(point_i) == np.ndarray:
            print point_i.shape, 'locs_shape'

        kneighs = self._get_info_i(point_i, info_i)
        point_i = self._prepare_input(point_i, kr)
        res = self.retriever[kr].query(point_i, int(kneighs), True)
        res = res[1], self._apply_preprocess_relative_pos(list(res[0]))
        ## Correct for another relative spatial measure (Save time indexing)
        res = self._apply_relative_pos_spec(res, point_i)
        return res

    ########################### Auxiliar functions ############################
    def _preformat_neighs_info(self, format_level, type_neighs,
                               type_sp_rel_pos):
        """Over-writtable function."""
        format_level, type_neighs, type_sp_rel_pos = 2, 'array', 'array'
        return format_level, type_neighs, type_sp_rel_pos

    def _check_proper_retriever(self):
        "Check the correctness of the retriever for this class."
        pass
#        try:
#            for k in range(self.k_perturb):
#                _, kr = self._map_perturb(k)
#                loc = self.retriever[kr].data[0]
#                self.retriever[kr].query(loc, 2)
#            check = True
#        except:
#            check = False
#        return check

    def _define_retriever(self, locs, pars_ret=None):
        "Define a kdtree for retrieving neighbours."
        if pars_ret is not None:
            leafsize = int(pars_ret)
        else:
            leafsize = locs.shape[0]
            leafsize = locs.shape[0]/100 if leafsize > 1000 else leafsize
        retriever = KDTree(locs, leaf_size=leafsize)
        self.retriever.append(retriever)
        ## Preprocess setting
        if self._ifdistance:
            self._apply_preprocess_relative_pos =\
                self._apply_preprocess_relative_pos_dim
        else:
            self._apply_preprocess_relative_pos =\
                self._apply_preprocess_relative_pos_null

    def _get_loc_from_idx(self, i_loc, kr=0):
        """Specific interaction with the data stored in retriever object."""
        data_locs = []
        i_loc = [i_loc] if type(i_loc) not in [list, np.ndarray] else i_loc
        for i in i_loc:
            print i_loc
            data_locs.append(self.retriever[kr].data[i])
        data_locs = np.array(data_locs)
        return data_locs


################################ R disctance ##################################
###############################################################################
class CircRetriever(SpaceRetriever):
    "Circular retriever."
    _default_ret_val = 0.1
    constant_neighs = None
    preferable_input_idx = False
    auto_excluded = True
    output_array = False

    ###################### Retrieve functions candidates ######################
    def _retrieve_neighs_general_spec(self, point_i, radius_i,
                                      ifdistance=False, kr=0):
        """General function to retrieve neighs in the specific way we want."""
        point_i = self._prepare_input(point_i, kr)
        res = self.retriever[kr].query_radius(point_i, radius_i, ifdistance)
        if ifdistance:
            res = list(res[0]), list(res[1])
            res = res[0], self._apply_preprocess_relative_pos(res[1])
        else:
            res = list(res), None
        ## Correct for another relative spatial measure (Save time indexing)
        res = self._apply_relative_pos_spec(res, point_i)
        return res

    def _retrieve_neighs_constant_nodistance(self, point_i, info_i={}, kr=0):
        """Retrieve neighs not computing distance by default.

        Parameters
        ----------
        point_i: int
            the indice of the point_i.
        """
        radius = self._get_info_i(point_i, info_i)
        point_i = self._prepare_input(point_i, kr)
        res = self.retriever[kr].query(point_i, radius, False)
        res = list(res), None
        return res

    def _retrieve_neighs_constant_distance(self, point_i, info_i={}, kr=0):
        """Retrieve neighs computing distance by default.

        Parameters
        ----------
        point_i: int
            the indice of the point_i.
        """
        radius = self._get_info_i(point_i, info_i)
        point_i = self._prepare_input(point_i, kr)
        res = self.retriever[kr].query(point_i, radius, True)
        res = list(res[0]), self._apply_preprocess_relative_pos(list(res[1]))
        ## Correct for another relative spatial measure (Save time indexing)
        res = self._apply_relative_pos_spec(res, point_i)
        return res

    ########################### Auxiliar functions ############################
    def _preformat_neighs_info(self, format_level, type_neighs,
                               type_sp_rel_pos):
        """Over-writtable function."""
        format_level, type_neighs, type_sp_rel_pos = 2, 'list', 'list'
        return format_level, type_neighs, type_sp_rel_pos

    def _check_proper_retriever(self):
        """Check the correctness of the retriever for this class."""
        pass
#        try:
#            for k in range(self.k_perturb):
#                _, kr = self._map_perturb(k)
#                loc = self.retriever[kr].data[0]
#                self.retriever[kr].query_radius(loc, 0.5)
#            check = True
#        except:
#            check = False
#        return check

    def _define_retriever(self, locs, pars_ret=None):
        "Define a kdtree for retrieving neighbours."
        if pars_ret is not None:
            leafsize = int(pars_ret)
        else:
            leafsize = locs.shape[0]
            leafsize = locs.shape[0]/100 if leafsize > 1000 else leafsize
        retriever = KDTree(locs, leaf_size=leafsize)
        self.retriever.append(retriever)
        ## Preprocess setting
        if self._ifdistance:
            self._apply_preprocess_relative_pos =\
                self._apply_preprocess_relative_pos_dim
        else:
            self._apply_preprocess_relative_pos =\
                self._apply_preprocess_relative_pos_null

    def _get_loc_from_idx(self, i_loc, kr=0):
        """Specific interaction with the data stored in retriever object."""
        data_locs = []
        i_loc = [i_loc] if type(i_loc) not in [list, np.ndarray] else i_loc
        for i in i_loc:
            data_locs.append(self.retriever[kr].data[i])
        data_locs = np.array(data_locs)
        return data_locs


############################## Windows Neighbours #############################
###############################################################################
class WindowsRetriever(SpaceRetriever):
    """Class which contains a retriever of window neighbours for
    n-dimensional grid data.
    """
    _default_ret_val = {'l': 1, 'center': 0, 'excluded': False}
    constant_neighs = True
    preferable_input_idx = False
    auto_excluded = False
    ## WARNING
    output_array = None

    ######################## Retrieve-driven retrieve #########################
    def set_iter(self, info_ret=None, max_bunch=None):
        info_ret = self._default_ret_val if info_ret is None else info_ret
        max_bunch = len(self) if max_bunch is None else max_bunch
        self._info_ret = info_ret
        self._max_bunch = max_bunch

#    def __iter__(self):
#        ## Prepare iteration
#        bool_input_idx = True
#        self._format_preparators(bool_input_idx)
#        self._constant_ret = True
#        self._format_retriever_function(bool_input_idx)
#        ## Prepare indices
#        indices = split_parallel(np.arange(self._n0), self._max_bunch)
#        ## Iteration
#        for idxs in indices:
#            neighs = self.retrieve_neighs(idxs)
#            yield indices, neighs

    def __iter__(self):
        ## Prepare iteration
        bool_input_idx = True
        self._format_preparators(bool_input_idx)
        self._constant_ret = True
        self._format_retriever_function(bool_input_idx)
        ## Iteration
        for inds, neighs, rel_pos in windows_iteration():
            ### KS 
            ### Set neighs_info !!!    
            yield inds, neighs, rel_pos

    ###################### Retrieve functions candidates ######################
    def _retrieve_neighs_general_spec(self, element_i, pars_ret,
                                      ifdistance=False, kr=0):
        """Retrieve all the neighs in the window described by pars_ret."""
        ## Get loc
        loc_i = self._prepare_input(element_i, kr)
        assert(len(loc_i.shape) == 2)
        neighs_info = generate_grid_neighs_coord(loc_i, self._shape,
                                                 self._ndim, **pars_ret)
        neighs = self.retriever[kr].map2indices_iss(neighs_info[0])
        ## Compute neighs_info
        if ifdistance:
            neighs_info = neighs, neighs_info[1]
            ## Correct for another relative spatial measure(Save time indexing)
            if self.relative_pos is not None:
                neighs_info = self._apply_relative_pos(neighs_info[0],
                                                       element_i,
                                                       neighs_info[1])
        else:
            neighs_info = neighs, None
        return neighs_info

    def _retrieve_neighs_constant_nodistance(self, element_i, pars_ret, kr=0):
        """Retrieve neighs not computing distance by default."""
        pars_ret = self._get_info_i(element_i, pars_ret)
        loc_i = self._prepare_input(element_i, kr)
        assert(len(loc_i.shape) == 2)
        neighs_info = generate_grid_neighs_coord(loc_i, self._shape,
                                                 self._ndim, **pars_ret)
        neighs_info = self.retriever[kr].map2indices_iss(neighs_info[0])
        return neighs_info

    def _retrieve_neighs_constant_distance(self, element_i, pars_ret, kr=0):
        """Retrieve neighs computing distance by default."""
        pars_ret = self._get_info_i(element_i, pars_ret)
        loc_i = self._prepare_input(element_i, kr)
        assert(len(loc_i.shape) == 2)
        neighs_info = generate_grid_neighs_coord(loc_i, self._shape,
                                                 self._ndim, **pars_ret)
        aux_neigh = self.retriever[kr].map2indices_iss(neighs_info[0])
        neighs_info = aux_neigh, neighs_info[1]
        ## Correct for another relative spatial measure (Save time indexing)
        if self.relative_pos is not None:
            neighs_info = self._apply_relative_pos(neighs_info[0], element_i,
                                                   neighs_info[1])
        return neighs_info

    ########################### Auxiliar functions ############################
    def _preformat_neighs_info(self, format_level, type_neighs,
                               type_sp_rel_pos):
        """Over-writtable function."""
        format_level, type_neighs, type_sp_rel_pos = 2, 'list', 'list'
        return format_level, type_neighs, type_sp_rel_pos

#    ### TODO: TOMOVE (virtual data)
#    def get_loc_i(self, element_i, kr):
#        """Specific class function to get locations from input. Overwrite the
#        generic function.
#        """
#        if type(element_i) == int:
#            element_i = self.retriever[kr].map2locs(element_i)
#        return element_i

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

        map2indices, map2locs, WindRetriever = create_window_utils(shape_)

        ## Store in a correct format
        self.retriever.append(WindRetriever(shape_, map2indices, map2locs))
        self._limits = limits
        self._shape = shape_
        self._ndim = len(limits)
        self._virtual_data = True
