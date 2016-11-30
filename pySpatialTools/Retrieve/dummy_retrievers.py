
"""
DummyRetrievers
---------------
Special retrievers for testing. It is a dummy class which contains all the
features we want to test from retrievers.

"""

import numpy as np
from retrievers import BaseRetriever
from pySpatialTools.utils.util_classes import SpatialElementsCollection


###############################################################################
############################ Space-Based Retrievers ###########################
###############################################################################
class DummyRetriever(BaseRetriever):
    """Dummy null retriever container. It gives the structure desired by the
    retrievers classes to work properly.
    """
    _default_ret_val = 0

    def __init__(self, n, autodata=True, input_map=None, output_map=None,
                 info_ret=None, info_f=None, constant_info=None,
                 perturbations=None, autoexclude=None, ifdistance=True,
                 relative_pos=None, bool_input_idx=None, typeret='space',
                 preferable_input_idx=None, constant_neighs=True,
                 bool_listind=None, types='array', auto_excluded=True):
        """General definition of retriever.

        Parameters
        ----------
        n: int
            the number of elements to be retrieved.
        autodata: optional (default=True)
            the data to query for their neighborhoods.
        input_map: function or None (default=None)
            the map applied to the input queried.
        output_map: function or list or None (default=None)
            the maps applied to the result of the retieved task.
        info_ret: optional (default=None)
            the information to set the core-retriever or to define the
            neighborhood.
        info_f: function (default=None)
            information creation to query for neighborhood.
        constant_info: boolean (default=None)
            if we are going to use it statically, inputting the same type
            of inputs.
        perturbations: pst.BasePerturbation (default=None)
            perturbations applied to the spatial model.
        autoexclude: boolean or None (default=None)
            if we want to exclude the element from its neighborhood.
        ifdistance: boolean (default=True)
            if we want to retrieve the distance or the relative position.
        relative_pos: function or pst.BaseRelative_positioner (default=None)
            the relative position function or object.
        bool_input_idx: boolean or None (default=None)
            if the input are going to be always indices (True), always the
            whole spatial information (False) or we do not know or even
            there is going to be not always the same (None)
        typeret: optional ['space', 'network'] (default='space')
            type of retriever.
        preferable_input_idx: boolean or None (default=None)
            if the core-retriever prefers be input the indices or the whole
            spatial information of the elements to be retrieved their neighs.
        constant_neighs: boolean (default=True)
            if there are always the same number of inputs, once it is set the
            retriever parameters.
        bool_listind: boolean or None (default=None)
            if the core-retriever is able to retrieve the elements in list.
        types: str option ['array', 'listobject', 'object']
            the type the pool of spatial elements are represented.
        auto_excluded: boolean (default=True)
            if the core retriever is able to exclude the element from its
            neighborhood.

        """
        ## Special inputs
        locs, autolocs, pars_ret = self._spec_pars_parsing(n, autodata, types)
        ## Definition of class parameters
        self._static_class_parameters_def(preferable_input_idx, typeret,
                                          constant_neighs, bool_listind,
                                          auto_excluded)
        ## Reset globals
        self._initialization()
        # IO mappers
        self._format_maps(input_map, output_map)
        # Location information
        self._format_locs(locs, autolocs)
        ## Retrieve information
        self._define_retriever(locs, pars_ret)
        ## Info_ret mangement
        self._format_retriever_info(info_ret, info_f, constant_info)
        # Perturbations
        self._format_perturbation(perturbations)
        # Output information
        self._format_output_information(autoexclude, ifdistance, relative_pos)
        self._format_exclude(bool_input_idx, self.constant_neighs)
        ## Format retriever function
        self._format_retriever_function()
        self._format_getters(bool_input_idx)
        # Preparation input and output
        self._format_preparators(bool_input_idx)
        self._format_neighs_info(bool_input_idx)
        ## Assert properly formatted
        self.assert_correctness()

    ###################### Class instantiation functions ######################
    def _static_class_parameters_def(self, preferable_input_idx, typeret,
                                     constant_neighs, bool_listind,
                                     auto_excluded):
        """The parameters are usually be static class parameters."""
        self.auto_excluded = auto_excluded
        self.preferable_input_idx = preferable_input_idx
        self.typeret = typeret
        self.constant_neighs = constant_neighs
        self.bool_listind = bool_listind

    def _spec_pars_parsing(self, n, autodata, types):
        """Parsing the specific specific parameters input."""
        ## Locs dummy definition
        locs = np.arange(n).reshape((n, 1))
        if types == 'list':
            locs = list(locs)
        elif types == 'listobject':
            locs = [DummyLocObject(e) for e in locs]
        elif types == 'object':
            locs = SpatialElementsCollection(list(locs))
        ## Autolocs and pars
        if autodata is True:
            autolocs = locs
        else:
            autolocs = autodata
            if autolocs is not None:
                if types == 'list':
                    autolocs = list(autolocs)
                elif types == 'listobject':
                    autolocs = [DummyLocObject(e) for e in autolocs]
                elif types == 'object':
                    autolocs = SpatialElementsCollection(list(autolocs))
        pars_ret = None
        return locs, autolocs, pars_ret

    def _define_retriever(self, locs, pars_ret=None):
        """Define a kdtree for retrieving neighbours.

        Parameters
        ----------
        locs: list, np.ndarray, or others
            spatial information of the whole pool of retrievable spatial
            elements.
        pars_ret: int or None (default)
            the parameters to set the core-retriever. In sklearn-KDTree
            core-retriever, we only need leafsize parameter.

        """
        class DummyAuxRet:
            def __init__(self, data):
                self.data = data
        if type(locs) == np.ndarray:
            locs = locs.astype(int)
        self.retriever.append(DummyAuxRet(locs))

    def _format_locs(self, locs, autolocs):
        self.data = None if autolocs is None else autolocs
        self._autodata = True if self.data is None else False

    def _preformat_neighs_info(self, format_level, type_neighs,
                               type_sp_rel_pos):
        """Over-writtable function. It is a function that given some of the
        properties of how the core-retriever is going to give us the
        information of the neighborhood.

        Parameters
        ----------
        format_level: int
            the level of information which gives neighborhood (see
            pst.Neighs_Info)
        type_neighs: str (optional)
            the type of neighs is given by the core-retriever (see
            pst.Neighs_Info)
        type_sp_rel_pos: str (optional)
            the type of relative position information is given by the
            core-retriever (see pst.Neighs_Info)

        Returns
        -------
        format_level: int
            the level of information which gives neighborhood (see
            pst.Neighs_Info)
        type_neighs: str (optional)
            the type of neighs is given by the core-retriever (see
            pst.Neighs_Info)
        type_sp_rel_pos: str (optional)
            the type of relative position information is given by the
            core-retriever (see pst.Neighs_Info)

        """
        format_level = 2
        if self.constant_neighs:
            type_neighs, type_sp_rel_pos = 'array', 'array'
        else:
            type_neighs, type_sp_rel_pos = 'list', 'list'
        return format_level, type_neighs, type_sp_rel_pos

    ######################### Needed getter functions #########################
    def _get_loc_from_idx(self, i):
        """Not list indexable interaction with data.

        Parameters
        ----------
        i: int
            the index of the element we want to get.
        kr: int (default = 0)
            the indice of the core-retriever selected. When there are location
            perturbations, the core-retriever it is replicated for each
            perturbation, so we need to select perturbated retriever. `kr`
            could be equal to the `k` or not depending on the type of
            perturbations.

        Returns
        -------
        loc_i: np.ndarray
            the spatial information of the element `i`.

        """
#        print i, kr
#        loc_i = self._get_loc_from_idx_indata(i)
#        print i, type(self.data_input), '0'*10
        if type(i) in [int, np.int32, np.int64]:
            loc_i = self.data_input[i]
        else:
            loc_i = []
            for j in i:
                loc_i.append(self.data_input[j])
        ## Same structure as input data
        if type(self.data_input) == np.ndarray:
            loc_i = np.array(loc_i)
#        print loc_i, 'm'*10, type(loc_i), type(self.data_input)
        return loc_i

    def _get_idx_from_loc(self, loc_i, kr=0):
        """Get indices from locations.

        Parameters
        ----------
        loc_i: np.ndarray
            the spatial information of the element `i`.
        kr: int (default = 0)
            the indice of the core-retriever selected. When there are location
            perturbations, the core-retriever it is replicated for each
            perturbation, so we need to select perturbated retriever. `kr`
            could be equal to the `k` or not depending on the type of
            perturbations.

        Returns
        -------
        indices: list
            the list of element we want to get from the data.

        """
#        print loc_i, self.retriever[kr].data.shape, type(loc_i)
        indices = []
        if self.bool_listind:
            for i in range(len(loc_i)):
                logi = np.where(self.retriever[kr].data == loc_i[i])
                if len(logi):
                    indices += list(logi[0])
        else:
            for j in range(len(self.retriever[kr].data)):
                if self.retriever[kr].data[j] == loc_i:
                    indices.append(j)
#            logi = np.where(self.retriever[kr].data == loc_i)
#            if len(logi):
#                indices = list(logi[0])
        return indices

    ######################### Format output functions #########################
    def _format_output_exclude(self, i_locs, neighs, dists, output=0, kr=0):
        """Format output with excluding.

        Parameters
        ----------
        i_loc: int, list or np.ndarray or other
            the information of the element (as index or the whole spatial
            information of the element to retieve its neighborhood)
        neighs: list of np.ndarray or np.ndarray
            the neighs indices for each iss in i_loc.
        dists: list of list of np.ndarray or np.ndarray
            the information or relative position in respect to each iss
        output: int (default = 0)
            the number of output mapper function selected.
        kr: int (default = 0)
            the indice of the core-retriever selected. When there are location
            perturbations, the core-retriever it is replicated for each
            perturbation, so we need to select perturbated retriever. `kr`
            could be equal to the `k` or not depending on the type of
            perturbations.

        Returns
        -------
        neighs: list of np.ndarray or np.ndarray
            the neighs indices for each iss in i_loc.
        dists: list of list of np.ndarray or np.ndarray
            the information or relative position in respect to each iss

        """
        neighs, dists = self._output_map[output](self, i_locs, (neighs, dists))
        neighs, dists = self._exclude_auto(i_locs, neighs, dists, kr)
        return neighs, dists

    def _format_output_noexclude(self, i_locs, neighs, dists, output=0, kr=0):
        """Format output without excluding the same i.

        Parameters
        ----------
        i_loc: int, list or np.ndarray or other
            the information of the element (as index or the whole spatial
            information of the element to retieve its neighborhood)
        neighs: list of np.ndarray or np.ndarray
            the neighs indices for each iss in i_loc.
        dists: list of list of np.ndarray or np.ndarray
            the information or relative position in respect to each iss
        output: int (default = 0)
            the number of output mapper function selected.
        kr: int (default = 0)
            the indice of the core-retriever selected. When there are location
            perturbations, the core-retriever it is replicated for each
            perturbation, so we need to select perturbated retriever. `kr`
            could be equal to the `k` or not depending on the type of
            perturbations.

        Returns
        -------
        neighs: list of np.ndarray or np.ndarray
            the neighs indices for each iss in i_loc.
        dists: list of list of np.ndarray or np.ndarray
            the information or relative position in respect to each iss

        """

        neighs, dists = self._output_map[output](self, i_locs, (neighs, dists))
        return neighs, dists

    ########################### Retriever functions ###########################
    def _retrieve_neighs_general_spec(self, point_i, info_i, ifdistance=True,
                                      kr=0):
        """General function to retrieve neighs in the specific way we want.

        Parameters
        ----------
        point_i: int
            the indice of the point_i.
        info_i: int (default = {})
            the information which defines the retrieved neighborhood regarding
            the selected model of neighborhood.
        ifdistance: boolean or None (default)
            if we want to retrieve distances.
        kr: int (default = 0)
            the indice of the core-retriever selected. When there are location
            perturbations, the core-retriever it is replicated for each
            perturbation, so we need to select perturbated retriever. `kr`
            could be equal to the `k` or not depending on the type of
            perturbations.

        Returns
        -------
        neighs_info: tuple (neighs, dists)
            the neighbourhood information prepared to be stored.

        """
        if ifdistance or ifdistance is None:
            neighs_info =\
                self._retrieve_neighs_constant_distance(point_i, info_i, kr)
        else:
            neighs_info =\
                self._retrieve_neighs_constant_nodistance(point_i, info_i, kr)
        return neighs_info

    def _retrieve_neighs_constant_nodistance(self, point_i, info_i={}, kr=0):
        """Retrieve neighs not computing distance by default.

        Parameters
        ----------
        point_i: int
            the indice of the point_i.
        info_i: int (default = {})
            the information which defines the retrieved neighborhood regarding
            the selected model of neighborhood.
        kr: int (default = 0)
            the indice of the core-retriever selected. When there are location
            perturbations, the core-retriever it is replicated for each
            perturbation, so we need to select perturbated retriever. `kr`
            could be equal to the `k` or not depending on the type of
            perturbations.

        Returns
        -------
        neighs_info: tuple (neighs, dists)
            the neighbourhood information prepared to be stored.

        """
        info_i = self._get_info_i(point_i, info_i)
        point_i = self._prepare_input(point_i, kr)
        ## Transformation to a list of arrays
        if self.preferable_input_idx:
            assert(type(point_i[0]) in [int, np.int32, np.int64])
            neighs = [self.retriever[kr].data[p] for p in point_i]
            if type(neighs[0]) != np.ndarray:
                neighs = [e.location for e in neighs]
            assert(type(neighs[0][0]) in [int, np.int32, np.int64])
        else:
            neighs = [p for p in point_i]
            if type(neighs[0]) != np.ndarray:
                neighs = [e.location for e in neighs]
            assert(type(neighs[0][0]) in [int, np.int32, np.int64])
        dists = None
        ## Constant neighs
        if self.constant_neighs:
            neighs = np.array(neighs)
        assert(len(point_i) == len(neighs))
        assert(type(neighs[0]) == np.ndarray)
        assert(type(neighs[0][0]) in [int, np.int32, np.int64])
        return neighs, dists

    def _retrieve_neighs_constant_distance(self, point_i, info_i={}, kr=0):
        """Retrieve neighs not computing distance by default.

        Parameters
        ----------
        point_i: int
            the indice of the point_i.
        info_i: int (default = {})
            the information which defines the retrieved neighborhood regarding
            the selected model of neighborhood.
        kr: int (default = 0)
            the indice of the core-retriever selected. When there are location
            perturbations, the core-retriever it is replicated for each
            perturbation, so we need to select perturbated retriever. `kr`
            could be equal to the `k` or not depending on the type of
            perturbations.

        Returns
        -------
        neighs_info: tuple (neighs, dists)
            the neighbourhood information prepared to be stored.

        """
        ## Retrieving neighs
        neighs, _ =\
            self._retrieve_neighs_constant_nodistance(point_i, info_i, kr)
        dists = [np.zeros((len(e), 1)) for e in neighs]
        if self.constant_neighs:
            neighs = np.array(neighs)
            dists = np.array(dists)
        neighs_info = neighs, dists
        ## Correct for another relative spatial measure (Save time indexing)
        point_i = self._prepare_input(point_i, kr)
        neighs_info = self._apply_relative_pos_spec(neighs_info, point_i)
        return neighs_info


class DummyLocObject:
    """Dummy location object to test location objects retrieving."""
    def __init__(self, information):
        self.location = information

    def __eq__(self, array):
        return np.all(self.location == array)

    def __iter__(self):
        for i in range(1):
            yield self.location
