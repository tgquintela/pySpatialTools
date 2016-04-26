
"""
DummyRetrievers
---------------
Special retrievers for testing.
"""

import numpy as np
from retrievers import Retriever


###############################################################################
############################ Space-Based Retrievers ###########################
###############################################################################
class DummyRetriever(Retriever):
    """Dummy null retriever container. It gives the structure desired by the
    retrievers classes to work properly.
    """
    _default_ret_val = 0

    def __init__(self, n, autodata=False, input_map=None, output_map=None,
                 info_ret=None, info_f=None, constant_info=None,
                 perturbations=None, autoexclude=None, ifdistance=None,
                 relative_pos=None, bool_input_idx=None, typeret='space',
                 preferable_input_idx=None, constant_neighs=True,
                 bool_listind=None):
        ## Special inputs
        locs, autolocs, pars_ret = self._spec_pars_parsing(n, autodata)
        ## Definition of class parameters
        self._static_class_parameters_def(preferable_input_idx, typeret,
                                          constant_neighs, bool_listind)
        ## Reset globals
        self._initialization()
        # IO mappers
        self._format_maps(input_map, output_map)
        ## Info_ret mangement
        self._format_retriever_info(info_ret, info_f, constant_info)
        # Location information
        self._format_locs(locs, autolocs)
        ## Retrieve information
        self._define_retriever(locs, pars_ret)
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
                                     constant_neighs, bool_listind):
        """The parameters are usually be static class parameters."""
        r = np.random.randint(0, 3)
        pos = [True, False, None]
        self.auto_excluded = pos[r]
        self.preferable_input_idx = preferable_input_idx
        self.typeret = typeret
        self.constant_neighs = constant_neighs
        self.bool_listind = bool_listind

    def _spec_pars_parsing(self, n, autodata):
        """Parsing the specific specific parameters input."""
        ## Locs dummy definition
        locs = np.arange(n).reshape((n, 1))
        ## Random list data class
#        r = np.random.randint(0, 2)
#        if r:
#            locs = [e for e in locs]
        ## Autolocs and pars
        autolocs = locs if autodata is True else autodata
        pars_ret = None
        return locs, autolocs, pars_ret

    def _define_retriever(self, locs, pars_ret=None):
        class DummyAuxRet:
            def __init__(self, data):
                self.data = data
        self.retriever.append(DummyAuxRet(locs))

    def _format_locs(self, locs, autolocs):
        self.data = None if autolocs is None else autolocs
        self._autodata = True if self.data is None else False

    def _preformat_neighs_info(self, format_level, type_neighs,
                               type_sp_rel_pos):
        """Over-writtable function."""
        format_level = 2
        if self.constant_neighs:
            type_neighs, type_sp_rel_pos = 'array', 'array'
        else:
            type_neighs, type_sp_rel_pos = 'list', 'list'
        return format_level, type_neighs, type_sp_rel_pos

    ######################### Needed getter functions #########################
    def _get_loc_from_idx(self, i, kr=0):
        """Not list indexable interaction with data."""
#        print i, kr
        loc_i = np.array(self.retriever[kr].data[i])
        return loc_i

    def _get_idx_from_loc(self, loc_i, kr=0):
        """Get indices from locations."""
#        print loc_i, self.retriever[kr].data.shape, type(loc_i)
        indices = []
        for i in range(len(loc_i)):
            indices += list(np.where(self.retriever[kr].data == loc_i[i])[0])
        return indices

    ######################### Format output functions #########################
    def _format_output_exclude(self, i_locs, neighs, dists, output=0, kr=0):
        "Format output with excluding."
        neighs, dists = self._output_map[output](self, i_locs, (neighs, dists))
        neighs, dists = self._exclude_auto(i_locs, neighs, dists, kr)
        return neighs, dists

    def _format_output_noexclude(self, i_locs, neighs, dists, output=0, kr=0):
        "Format output without excluding the same i."
        neighs, dists = self._output_map[output](self, i_locs, (neighs, dists))
        return neighs, dists

    ########################### Retriever functions ###########################
    def _retrieve_neighs_general_spec(self, point_i, info_i, ifdistance=True,
                                      kr=0):
        """General function to retrieve neighs in the specific way we want."""
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
        """
        info_i = self._get_info_i(point_i, info_i)
        point_i = self._prepare_input(point_i, kr)
        ## Transformation to a list of arrays
        if self.preferable_input_idx:
            assert(type(point_i[0]) in [int, np.int32, np.int64])
            neighs = [self.data_input[p] for p in point_i]
            assert(type(neighs[0][0]) in [int, np.int32, np.int64])
        else:
            neighs = [p for p in point_i]
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
