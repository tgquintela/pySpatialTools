
"""
Retrievers
----------
The objects to retrieve neighbours in flat (non-splitted) space or precomputed
mapped relations.
The retrievers can be defined in different topologic spaces.
This class acts as a wrapper to the core definition of this retrievers. In
this class are coded all the structure and administrative stuff in order to
manage and optimize the retrieve of neighbourhood.

Structure:
----------
* Retrievers
- Retrievers
    * Implicit Retrievers
    - SpatialRetrievers
        - KDTreeBasedRetrievers
            - KRetriever
            - RadiusRetriever
        - WindowRetriever
    * Explicit Retrievers
    - NetworkRetrievers
        - DirectMapping
        - OrderRetriever
        - MaxDistanceRetriever

Main Functionalities
--------------------
Function to retrieve required information to retrieve neighbourhood:

get_loc_i: function which is useful to get locations from the data input
    pool using locations itself or indices.
get_indice_i: function which serves to get indices of the pool of data
    locations available to retrieve.
_prepare_input: internal function useful to prepare the input locations
    information in order to ensure that the core-retriever gets the
    information it needs in order to retrieve properly neighbourhoods.
_get_info_i: function which serves to get the retrieving information of
    element_i to retrieve its neighbourhood. If there is an open system
    it uses other information as the own location information to define
    the retrieving information.

Functions to retrieve neighbourhood:
retrieve_neighs: the main function to retrieve the neighbourhood. It
    accepts as parameters the loc_i (information of the element i of
    which we want to get its neighbourhood), info_i (retrieving
    information), 

Functions to format neighbourhood information
_exclude_auto:



TODO:
----
- Ifdistance better implementation
- Exclude better implementation
- Multiple regions
- Multiple points to get neighs
- SpatialElementsCollection support
- Remove Locations support

"""

import numpy as np
import warnings
from copy import copy
from scipy.sparse import coo_matrix
from aux_retriever import _check_retriever, _general_autoexclude,\
    _array_autoexclude, _list_autoexclude
from ..utils import NonePerturbation
from ..utils import ret_filter_perturbations
from ..utils.util_classes import SpatialElementsCollection, Neighs_Info

arraytypes = [np.ndarray, list]
inttypes = [int, np.int32, np.int64]


class Retriever:
    """Class which contains the retriever of elements.
    """
    __name__ = 'pySpatialTools.Retriever'

    ######################## Retrieve-driven retrieve #########################
    def set_iter(self, info_ret=None, max_bunch=None):
        info_ret = self._default_ret_val if info_ret is None else info_ret
        max_bunch = len(self) if max_bunch is None else max_bunch
        self._info_ret = info_ret
        self._max_bunch = max_bunch

    def __iter__(self):
        ## Prepare iteration
        bool_input_idx, constant_info = True, True
        self._format_general_information(bool_input_idx, constant_info)
        # Input indices
        ## Iteration
        for i in range(self._n0):
            iss = self.get_indice_i(i)
            neighs = self.retrieve_neighs(i)
            yield iss, neighs

    ##################### Retrieve candidates functions #######################
    def _retrieve_neighs_static(self, i_loc, output=0):
        """Retrieve neighs and distances. This function acts as a wrapper to
        more specific functions designed in the specific classes and methods.
        This function is composed by mutable functions in order to take profit
        of saving times excluding flags. It assumes staticneighs.
        """
        ## 1. Retrieve neighs
        neighs, dists = self._retrieve_neighs_spec(i_loc, {})
        ## 2. Format output
#        print 'setting000:', i_loc, neighs, dists, self._retrieve_neighs_spec
        neighs_info = self._format_output(i_loc, neighs, dists, output)
        ## 3. Format neighs_info
        self.neighs_info._reset_stored()
#        print 'setting:', i_loc, neighs_info, type(dists), dists, self._ifdistance, type(neighs_info[0])
        self.neighs_info.set(neighs_info, self.get_indice_i(i_loc))
        assert(self.staticneighs == self.neighs_info.staticneighs)
        assert(self.staticneighs)
        neighs_info = self.neighs_info.copy()
        print neighs_info.set_ks, neighs_info.ks, range(self.k_perturb+1), self.neighs_info.ks
        neighs_info.set_ks(range(self.k_perturb+1))
        print neighs_info.ks, range(self.k_perturb+1), self.neighs_info.ks
        assert(neighs_info.ks == range(self.k_perturb+1))
        return neighs_info

    def _retrieve_neighs_dynamic(self, i_loc, output=0):
        """Retrieve neighs and distances. This function acts as a wrapper to
        more specific functions designed in the specific classes and methods.
        This function is composed by mutable functions in order to take profit
        of saving times excluding flags. It assumes different preset
        perturbations to retrieve.
        """
        neighs_info = []
        ks = list(range(self.k_perturb+1))
        for k in ks:
            ## 1. Map perturb
            _, k_r = self._map_perturb(k)
            ## 2. Retrieve neighs
            neighs, dists = self._retrieve_neighs_spec(i_loc, {}, kr=k_r)
            nei_k = self._format_output(i_loc, neighs, dists, output, kr=k_r)
            neighs_info.append(nei_k)
        ## 3. Format neighs_info
        self.neighs_info._reset_stored()
#        print neighs_info, '1'*50, self.neighs_info.format_set_info
        self.neighs_info.set((neighs_info, ks), self.get_indice_i(i_loc))
#        print self.staticneighs, self.neighs_info.staticneighs
        assert(self.staticneighs == self.neighs_info.staticneighs)
        neighs_info = self.neighs_info.copy()
        assert(neighs_info.ks == range(self.k_perturb+1))
        return neighs_info

    def _retrieve_neighs_general(self, i_loc, info_i={}, ifdistance=None,
                                 k=None, output=0):
        """Retrieve neighs and distances. This function acts as a wrapper to
        more specific functions designed in the specific classes and methods.
        """
        assert(not self._constant_ret)
        ## 0. Prepare variables
        info_i, ifdistance, ks =\
            self._format_inputs_retriever(i_loc, info_i, ifdistance, k, output)
        ## 1. Retrieve neighs
        if self.staticneighs:
            assert(self.staticneighs == self.neighs_info.staticneighs)

            neighs, dists = self._retrieve_neighs_spec(i_loc, info_i,
                                                       ifdistance)
            ## 2. Format output
            neighs_info = self._format_output(i_loc, neighs, dists, output)
            ## Store
            self.neighs_info._reset_stored()
            self.neighs_info.set(([neighs_info], ks), self.get_indice_i(i_loc))
            neighs_info = self.neighs_info.copy()

#            # Get neighs info
##            print 'b0'*20, i_loc, info_i, ifdistance
#            neighs, dists =\
#                self._retrieve_neighs_spec(i_loc, info_i, ifdistance)
##            print 'b'*25, neighs, dists
#            ## 2. Format output
#            neighs_info = self._format_output(i_loc, neighs, dists, output)
#            neighs_info = [neighs_info]
######## To use in future
        else:
            neighs_info = []
            for k in range(len(ks)):
                # Map perturb
                _, k_r = self._map_perturb(k)
                # Retrieve
                neighs, dists =\
                    self._retrieve_neighs_spec(i_loc, info_i, ifdistance, k_r)
                # Format output
                nei_k = self._format_output(i_loc, neighs, dists, output, k_r)
                neighs_info.append(nei_k)
        ## 3. Format neighs_info
#        print 'a'*100, i_loc, self.k_perturb, neighs_info, type(neighs_info[0])
#        print self.staticneighs, ks == 0, self.get_indice_i(i_loc)
#        print self.neighs_info.staticneighs, self.neighs_info._set_iss
#        print self.neighs_info._set_info, self.neighs_info.set_neighs
            self.neighs_info._reset_stored()
            self.neighs_info.set((neighs_info, ks), self.get_indice_i(i_loc))
            assert(self.staticneighs == self.neighs_info.staticneighs)
            neighs_info = self.neighs_info.copy()
            if self.staticneighs:
                neighs_info.set_ks(range(self.k_perturb+1))
                assert(neighs_info.ks == range(self.k_perturb+1))
            else:
                print neighs_info.ks, range(self.k_perturb+1), ks
                assert(neighs_info.ks == ks)
        return neighs_info

    def _format_inputs_retriever(self, i_loc, info_i, ifdistance, k, output):
        """Format inputs retriever check and format the inputs for retrieving.
        """
        # Prepare information retrieve
#        print 'input', info_i
        info_i = self._get_info_i(i_loc, info_i)
#        print 'output', info_i
        #i_loc = self._get_loc_i(i_loc)
        ifdistance = self._ifdistance if ifdistance is None else ifdistance
        # Prepare perturbation index
        ks = [k] if type(k) == int else k
        ks = range(self.k_perturb+1) if ks is None else ks
        ## Check output (TODO)
        return info_i, ifdistance, ks

    ######################### Perturbation management #########################
    ###########################################################################
    def add_perturbations(self, perturbations):
        """Add perturbations."""
        perturbations = ret_filter_perturbations(perturbations)
        assert(type(perturbations) == list)
        for p in perturbations:
            self._dim_perturb.append(p.k_perturb)
            self._perturbators.append(p)
            self._create_map_perturbation()
            self._add_perturbated_retrievers(p)
        ## Reformat retriever functions
        self._format_retriever_function()
        self._format_neighs_info(self.bool_input_idx)

    def _format_perturbation(self, perturbations):
        """Format initial perturbations."""
        if perturbations is None:
            def _map_perturb(x):
                if x != 0:
                    raise IndexError("Not perturbation available.")
                return 0, 0
            self._map_perturb = _map_perturb
            self._dim_perturb = [1]
        else:
            self.add_perturbations(perturbations)

    def _create_map_perturbation(self):
        """Create the map for getting the perturbation object.
        The first inde of the map_perturb is an index in the _perturbators,
        the other one is an index for the retrievers.
        """
        ## 0. Creation of the mapper array
        limits = np.cumsum([0] + list(self._dim_perturb))
        sl = [slice(limits[i], limits[i+1]) for i in range(len(limits)-1)]
        ## Build a mapper
        mapper = np.zeros((np.sum(self._dim_perturb), 2)).astype(int)
        for i in range(len(sl)):
            inds = np.zeros((sl[i].stop-sl[i].start, 2))
            inds[:, 0] = i
            if self._perturbators[i]._perturbtype != 'none':
                ## TODO: perturbations of aggregations (networks)
                if self.typeret == 'space':
                    max_n = mapper[:, 1].max() + 1
                    inds[:, 1] = np.arange(sl[i].stop-sl[i].start) + max_n
            mapper[sl[i]] = inds

        ## 1. Creation of the mapper function
        def map_perturb(x):
            if x < 0:
                raise IndexError("Negative numbers can not be indices.")
            if x > self.k_perturb:
                msg = "Out of bounds. There are only %s perturbations."
                raise IndexError(msg % str(self.k_perturb))
            return mapper[x]
        ## 2. Storing mapper function
        self._map_perturb = map_perturb

    def _add_perturbated_retrievers(self, perturbation):
        """Add a perturbated retriever in self.retriever using the class
        function self._define_retriever."""
        if perturbation._categorytype == 'location':
            self.staticneighs = False
            self._format_neighs_info(self.bool_input_idx)
            for k in range(perturbation.k_perturb):
                locs_p = perturbation.apply2locs(self.retriever[0].data, k=k)
                if type(locs_p) == np.ndarray:
                    self._define_retriever(locs_p[:, :, 0])
                else:
                    if type(self.retriever[0].data) == list:
                        self._define_retriever(locs_p[0])
                    else:
                        aux_locs = SpatialElementsCollection(locs_p[0])
                        self._define_retriever(aux_locs)

    @property
    def k_perturb(self):
        return np.sum(self._dim_perturb)-1

    ######################### Aggregation management ##########################
    ###########################################################################

    ########################### Auxiliar functions ############################
    ###########################################################################
    def _initialization(self):
        """Mutable class parameters reset."""
        ## Elements information
        self.data = None
        self._autodata = True
        self._virtual_data = False
        ## Retriever information
        self.retriever = []
        self._info_ret = None
        self._info_f = None
        self._max_bunch = 1
        ## Perturbation
        self._dim_perturb = [1]
        self._map_perturb = lambda x: (0, 0)
        self._perturbators = [NonePerturbation()]
        self.staticneighs = True
        ## External objects to apply
        self.relative_pos = None
        self._ndim_rel_pos = 1
        ## IO information
        self.bool_input_idx = None
        self._autoexclude = False
        self._ifdistance = False
        self._autoret = True
        self._heterogenous_input = False
        self._heterogenous_output = False
        ## Neighs info
        self.neighs_info = Neighs_Info(staticneighs=True)
        ## IO methods
        self._input_map = lambda s, i: i
        self._output_map = [lambda s, i, x: x]
        ## Check
        _check_retriever(self)

    def _format_general_information(self, bool_input_idx, constant_info):
        """Assumption parameters:
        - self._info_ret
        - self.k_perturb
        """
#        print '9'*15, self._ifdistance
        ## Retrieve information getters and functions
        self._format_retriever_info(self._info_ret, self._info_f,
                                    constant_info)
        self._format_retriever_function()
        ## Getters data
        self._format_getters(bool_input_idx)
        ## Format output functions
        self._format_exclude(bool_input_idx, self.constant_neighs)
        # Preparation input and output
        self._format_preparators(bool_input_idx)
        self._format_neighs_info(bool_input_idx)
#        print '9+'*15, self._ifdistance

    def assert_correctness(self):
        """Assert the class is formatted properly."""
        assert(self.staticneighs == self.neighs_info.staticneighs)

    ################################ Formatters ###############################
    ###########################################################################
    ## Main formatters of the class. They adapt the class to the problem
    ## in order to be efficient and do the proper work.
    ###################### Output information formatting ######################
    def _format_output_information(self, autoexclude, ifdistance, relativepos):
        """Format functions to use in output creation."""
        ## Autoexclude managing
        self._autoexclude = autoexclude
        if autoexclude:
            self._format_output = self._format_output_exclude
        else:
            self._format_output = self._format_output_noexclude
        ## Ifdistance managing
        self._ifdistance = ifdistance
        ## Relative position managing
        self.relative_pos = relativepos
        if relativepos is None:
            self._apply_relative_pos = self._dummy_relative_pos
            self._apply_relative_pos_spec = self._apply_relative_pos_null
        else:
            self._apply_relative_pos_spec = self._apply_relative_pos_complete
            self._apply_relative_pos = self._general_relative_pos

    def _format_exclude(self, bool_input_idx, constant_neighs):
        """Format the excluding auto elements."""
        ## Inputs
        if bool_input_idx is True:
            self._build_excluded_elements =\
                self._indices_build_excluded_elements
        elif bool_input_idx is False:
            self._build_excluded_elements = self._locs_build_excluded_elements
        else:
            self._build_excluded_elements =\
                self._general_build_excluded_elements
        ## Excluding neighs_info
        # Excluding or not
        excluding = False if self.auto_excluded else True
        excluding = excluding if self._autoexclude else False
        if excluding:
            self._exclude_auto = self._exclude_auto_general
            if constant_neighs is True:
                self._exclude_elements = _array_autoexclude
            elif constant_neighs is False:
                self._exclude_elements = _list_autoexclude
            else:
                self._exclude_elements = _general_autoexclude
        else:
            self._exclude_auto = self._null_exclude_auto

    ####################### Core interaction formatting #######################
    ## Format information retrieve getters
    def _format_retriever_info(self, info_ret, info_f, constant_info):
        """Format properly the retriever information."""
        if type(info_ret).__name__ == 'function':
            self._info_f = info_ret
            self._info_ret = self._default_ret_val
        else:
            self._info_f = info_f
            aux_default = self._default_ret_val
            self._info_ret = aux_default if info_ret is None else info_ret
        ## Constant retrieve?
        if constant_info:
            self._constant_ret = True
            if info_f is None:
                self._get_info_i = self._dummy_get_info_i_stored
                if '__len__' in dir(info_ret) and self.data_input is not None:
                    if len(info_ret) == len(self.data_input):
                        self._get_info_i = self._dummy_get_info_i_indexed
            else:
                self._get_info_i = self._dummy_get_info_i_f
        else:
            self._get_info_i = self._general_get_info_i
            self._constant_ret = False

    def _format_retriever_function(self):
        """Format function to retrieve. It defines the main retrieve functions.
        """
        ## Format Retrievers function
        if self._constant_ret:
            if not self._ifdistance:
                self._retrieve_neighs_spec =\
                    self._retrieve_neighs_constant_nodistance
            else:
                self._retrieve_neighs_spec =\
                    self._retrieve_neighs_constant_distance
            if self.k_perturb == 0 or self.staticneighs:
                self.retrieve_neighs = self._retrieve_neighs_static
            else:
                self.retrieve_neighs = self._retrieve_neighs_dynamic
        ## Format specific function
        else:
            self.retrieve_neighs = self._retrieve_neighs_general
            self._retrieve_neighs_spec = self._retrieve_neighs_general_spec

    def _format_getters(self, bool_input_idx):
        ## Format retrieve locs and indices
        self._format_get_loc_i(bool_input_idx)
        self._format_get_indice_i(bool_input_idx)

    ######################### Neighs_Info formatting ##########################
    ## Format the neighs information in order to optimize the parsing of the
    ## information by formatting the class Neighs_Info
    def _preformat_neighs_info(self, format_level, type_neighs,
                               type_sp_rel_pos):
        """Over-writtable function."""
        return format_level, type_neighs, type_sp_rel_pos

    def _format_neighs_info(self, bool_input_idx, format_level=None,
                            type_neighs=None, type_sp_rel_pos=None):
        """Format neighs_info object in order to have better improvement and
        robusticity in the program.
        """
        ## Preformatting neighs_info
        format_level, type_neighs, type_sp_rel_pos =\
            self._preformat_neighs_info(format_level, type_neighs,
                                        type_sp_rel_pos)
        ## Setting structure
        if self._constant_ret:
            ## If constant and static
            if self.k_perturb == 0 or self.staticneighs:
                    format_structure = 'tuple_only'
#                    format_structure = 'tuple_tuple'
            ## If constant and dynamic
            else:
                #format_structure = 'tuple_list_tuple_only'
                format_structure = 'tuple_list_tuple'
        ## If not constant
        else:
            format_structure = 'tuple_list_tuple'

        ## Setting iss
#        if bool_input_idx:
#            format_set_iss = 'general'
#        else:
#            format_set_iss = 'null'
        format_set_iss = 'list'

#        print 'poi'*20, type_neighs, type_sp_rel_pos, self.staticneighs, self.neighs_info.staticneighs
        ## Neighs info setting
        self.neighs_info = Neighs_Info(format_set_iss=format_set_iss,
                                       format_structure=format_structure,
                                       staticneighs=self.staticneighs,
                                       ifdistance=self._ifdistance,
                                       format_level=format_level,
                                       type_neighs=type_neighs,
                                       type_sp_rel_pos=type_sp_rel_pos)

    def set_neighs_info(self, type_neighs, type_sp_rel_pos):
        """Utility function in order to reset neighs_info types."""
        self.neighs_info.set_types(type_neighs, type_sp_rel_pos)

    ######################### Input-output formatting #########################
    ## Format the interactions with the data in order to get the indices or the
    ## required spatial elements and to adapt the input information to retrieve
    ## its neighbourhood.
    def _format_maps(self, input_map, output_map):
        if input_map is not None:
            self._input_map = input_map
        if output_map is not None:
            if type(output_map).__name__ == 'function':
                self.output_map = [output_map]
            else:
                assert(type(output_map) == list)
                assert(all([type(m).__name__ == 'function'
                            for m in output_map]))
                self._output_map = output_map

    def _format_preparators(self, bool_input_idx):
        """Format the prepare inputs function in order to be used properly and
        efficient avoiding extra useless computations.
        """
        if self.preferable_input_idx == bool_input_idx:
            self._prepare_input = self._dummy_prepare_input
        elif bool_input_idx is None:
            self._prepare_input = self._general_prepare_input
        elif self.preferable_input_idx:
            self._prepare_input = self._dummy_loc2idx_prepare_input
        elif not self.preferable_input_idx:
            self._prepare_input = self._dummy_idx2loc_prepare_input

    def _format_get_loc_i(self, bool_input_idx):
        """Format the get locations function."""
        ## General functions
        if bool_input_idx is True:
            self.get_loc_i = self._get_loc_i_general_from_indices
        elif bool_input_idx is False:
            self.get_loc_i = self._get_loc_i_general_from_locations
        else:
            self.get_loc_i = self._get_loc_i_general
        ## Format direct interactors with the data
        if self._autodata is True:
            if self.bool_listind is True:
                self._get_loc_from_idxs = self._get_loc_from_idxs_listind
            else:
                self._get_loc_from_idxs = self._get_loc_from_idxs_notlistind
        else:
            self._get_loc_from_idx = self._get_loc_from_idx_indata
            if type(self.data_input) == np.ndarray:
                self._get_loc_from_idxs = self._get_loc_from_idxs_listind
            else:
                self._get_loc_from_idxs = self._get_loc_from_idxs_notlistind

    def _format_get_indice_i(self, bool_input_idx):
        """Format the get indice function."""
        ## General functions
        if bool_input_idx is True:
            self.get_indice_i = self._get_indice_i_general_from_indices
        elif bool_input_idx is False:
            self.get_indice_i = self._get_indice_i_general_from_locations
        else:
            self.get_indice_i = self._get_indice_i_general
        ## Format direct interactors with the data
        if self._autodata is True:
            if self.bool_listind is True:
                self._get_idxs_from_locs = self._get_idxs_from_locs_listind
            else:
                self._get_idxs_from_locs = self._get_idxs_from_locs_notlistind
        else:
            self._get_idx_from_loc = self._get_idx_from_loc_indata
            self._get_idxs_from_locs = self._get_idxs_from_locs_notlistind
#            if type(self.data) == list:
#                self._get_idxs_from_locs = self._get_idxs_from_locs_notlistind
#            else:
#                self._get_idxs_from_locs = self._get_idxs_from_locs_listind

    ###########################################################################
    ################################ Auxiliar #################################
    ###########################################################################
    ############################### Exclude auto ##############################
    ## Collapse to _exclude_auto in _format_exclude
    # Returns
    # -------
    # neighs: list of np.ndarray or np.ndarray
    #     the neighs for each iss in i_loc
    # dists: list of list of np.ndarray or np.ndarray
    #     the information or relative position in respect to each iss in i_loc
    #
    def _exclude_auto_general(self, i_loc, neighs, dists, kr=0):
        """Exclude auto elements if there exist in the neighs retrieved.
        This is a generic function independent on the type of the element.
        """
        ## 0. Detect input i_loc and retrieve to_exclude_elements list
#        print '=0'*15, i_loc, neighs, type(i_loc), len(neighs), self._build_excluded_elements
        to_exclude_elements = self._build_excluded_elements(i_loc, kr)
        ## 1. Excluding task
#        print 'point of pre debug', neighs, dists, to_exclude_elements
        neighs, dists =\
            self._exclude_elements(to_exclude_elements, neighs, dists)
#        print 'point of shit debug', neighs, dists, self._exclude_elements
#        assert(len(neighs) != 0)
        return neighs, dists

    def _null_exclude_auto(self, i_loc, neighs, dists, kr=0):
        return neighs, dists

    ############################# Exclude managing ############################
    ## Collapse to _build_excluded_elements in _format_exclude
    # Returns
    # -------
    # to_exclude_elements: list of list of ints
    #     the indices of the exclude elements.
    #
    def _general_build_excluded_elements(self, i_loc, kr=0):
        """Build the excluded points from i_loc if it is not an index.

        Parameters
        ----------
        i_loc: np.ndarray, shape(iss, dim) or shape(dim,)
            the locations we want to retrieve their neighbourhood.
        """
        # If it is an indice
        if type(i_loc) in [int, np.int32, np.int64, list]:
            to_exclude_elements =\
                self._indices_build_excluded_elements(i_loc, kr)
        # If it is an element spatial information
        else:
            to_exclude_elements =\
                self._locs_build_excluded_elements(i_loc, kr)
        return to_exclude_elements

    def _indices_build_excluded_elements(self, i_loc, kr=0):
        """
        Parameters
        ----------
        i_loc: list of ints or int
            the locations we want to retrieve their neighbourhood.
        """
        # If it is an indice
        if type(i_loc) in [int, np.int32, np.int64]:
            to_exclude_elements = self._int_build_excluded_elements(i_loc, kr)
        elif type(i_loc) == list:
            to_exclude_elements = self._list_build_excluded_elements(i_loc, kr)
        return to_exclude_elements

    def _locs_build_excluded_elements(self, i_loc, kr=0):
        """
        Parameters
        ----------
        i_loc: np.ndarray, shape(iss, dim) or shape(dim,)
            the locations we want to retrieve their neighbourhood.
        """
        ## 0. Preparing input
        i_loc = [i_loc] if type(i_loc) not in arraytypes else i_loc
        ## 1. Building indices to exclude
        to_exclude_elements = []
        for i in range(len(i_loc)):
            # Getting indices from the pool of elements
            to_exclude_elements.append(self.get_indice_i(i_loc[i]))
        ## 2. Check correct output
#        print to_exclude_elements, i_loc
#        assert(all([type(e) in arraytypes for e in to_exclude_elements]))
#        assert(all([all([type(e1) in inttypes for e1 in e])
#                    for e in to_exclude_elements if len(e)]))
        return to_exclude_elements

    def _list_build_excluded_elements(self, i_loc, kr=0):
        to_exclude_elements = [[i_loc[i]] for i in range(len(i_loc))]
        return to_exclude_elements

    def _int_build_excluded_elements(self, i_loc, kr=0):
        to_exclude_elements = [[i_loc]]
        return to_exclude_elements

    ############################# InfoRet managing ############################
    ## Collapse to _get_info_i in _format_retriever_info
    # Returns
    # -------
    # info_ret: optional
    #     the information parameters to retrieve neighbourhood.
    #
#    def _dummy_get_info_i(self, i_loc, info_i):
#        return info_i

    def _dummy_get_info_i_stored(self, i_loc, info_i=None):
        """Dummy get retrieve information."""
        return self._info_ret

    def _dummy_get_info_i_indexed(self, i_loc, info_i=None):
        """Dummy indexable retrieve information."""
        ### TODO: Referenced that __len__ and == len(self.data_input)
        return self._info_ret[i_loc]

    def _dummy_get_info_i_f(self, i_loc, info_i=None):
        """Dummy get retrieve information using function."""
        return self._info_f(i_loc, self._info_ret)

    def _general_get_info_i(self, i_loc, info_i):
        """Get retrieving information for each element i_loc. Comunicate the
        input i with the data_input. It is a generic function independent on
        the type of the elements we want to retrieve.
        """
        if info_i in [{}, [], None]:
            if self._info_f is None:
                info_i = self._dummy_get_info_i_stored(i_loc, info_i)
                logi = self.data_input is not None
                if '__len__' in dir(self._info_ret) and logi:
                    if len(self._info_ret) == len(self.data_input):
                        info_i = self._dummy_get_info_i_indexed(i_loc, info_i)
            else:
                info_i = self._dummy_get_info_i_f(i_loc, info_i)
        return info_i

    ########################### GetLocation managing ##########################
    ## Collapse to _prepare_input in _format_preparators
    # Returns
    # -------
    # spatial_info: list of ints or list of np.ndarray or np.ndarray
    #     the spatial information of required to retrieve neighbourhood from
    #     core retriever.
    #
    def _general_prepare_input(self, i_loc, kr=0):
        """General prepare input."""
        if self.preferable_input_idx:
            i_mloc = self._get_indice_i_general(i_loc, kr)
        else:
            i_mloc = self._get_loc_i_general(i_loc)
        return i_mloc

    def _dummy_prepare_input(self, i_loc, kr=0):
        """Dummy prepare input."""
        # Formatting to contain list of iss
        if not '__len__' in dir(i_loc):
            i_loc = [i_loc]
        if not self.preferable_input_idx:
            if type(i_loc) == np.ndarray:
                i_loc = list(i_loc) if len(i_loc.shape) == 2 else [i_loc]
            i_loc = [i_loc] if type(i_loc) != list else i_loc
        i_mloc = self._input_map(self, i_loc)
        return i_mloc

    def _dummy_idx2loc_prepare_input(self, i_loc, kr=0):
        """Dummy prepare input transforming indice to location."""
        # Formatting to contain list of iss
        if type(i_loc) not in [list, np.ndarray]:
            i_loc = [i_loc]
        i_loc = self._input_map(self, i_loc)
        loc_i = self.get_loc_i(i_loc)
        return loc_i

    def _dummy_loc2idx_prepare_input(self, loc_i, kr=0):
        """Dummy prepare input transforming location to indice."""
#        loc_i = np.array(loc_i)
#        if len(loc_i.shape) == 1:
#            loc_i = loc_i.reshape((1, len(loc_i)))
        ## Preprocessing to have list of locs
        if type(loc_i) != list:
            if type(loc_i) == np.ndarray:
                if len(loc_i.shape) == 1:
                    loc_i = loc_i.reshape((1, len(loc_i)))
            else:
                loc_i = [loc_i]
        loc_i = self._input_map(self, loc_i)
        i_loc = self.get_indice_i(loc_i, kr)
        return i_loc

    ############################## Get locations ##############################
    ## Collapse to get_loc_i in _format_get_loc_i
    # Get locations from the input data or the external loc data input.
    # Returns
    # -------
    # locations: list of np.ndarray or np.ndarray
    #     the spatial information of storage from the indications input.
    #
    ### Standart output
    # SAME type as input data
    #####################################################################
    def _get_loc_from_idxs_notlistind(self, i_loc):
        """Specific interaction with the data stored in retriever object."""
        data_locs = []
#        print i_loc, type(i_loc), 'p'*10, type(self.data_input)
        i_loc = [i_loc] if type(i_loc) not in arraytypes else i_loc
        for i in i_loc:
#            print i, self._get_loc_from_idx
            data_locs.append(self._get_loc_from_idx(i))
        ## Same structure as input data
        if type(self.data_input) == np.ndarray:
            data_locs = np.array(data_locs)
        return data_locs

    def _get_loc_from_idxs_listind(self, i_loc):
        """Specific interaction with the data stored in retriever object."""
        locs_i = self._get_loc_from_idx(i_loc)
#        print 'a'*10, locs_i, type(locs_i), type(locs_i[0]), type(self.data_input)
        ## Same structure as input data
        if type(self.data_input) == np.ndarray:
            locs_i = np.array(locs_i)
        return locs_i

    def _get_loc_from_idx_indata(self, i_loc):
        """Get data from indata."""
#        i_loc = i_loc if type(i_loc) in [np.ndarray, list] else [i_loc]
        if type(i_loc) in arraytypes:
            locs_i = [self.data_input[i] for i in i_loc]
            ## Same structure as input data
            if type(self.data_input) == np.ndarray:
                locs_i = np.array(locs_i)
        else:
            locs_i = self.data_input[i_loc]
        return locs_i

    def _get_loc_i_general_from_locations(self, i_loc):
        """Get element spatial information from spatial information.
        Format properly the input spatial information."""
        ### TO CHECK
#        print 'o'*20, i_loc, self.data_input
        ## Preprocessing to have list of locs
        if type(i_loc) != list:
            if type(i_loc) == np.ndarray:
                if len(i_loc.shape) == 1:
                    i_loc = i_loc.reshape((1, len(i_loc)))
            else:
                i_loc = [i_loc]
        #####################################################################
        if type(i_loc) == list:
            if len(i_loc) == 0:
                return i_loc
            if type(i_loc[0]) == np.ndarray:
                loc_i = np.array(i_loc)
            else:
                loc_i = i_loc
        elif type(i_loc) == np.ndarray:
            loc_i = self._get_loc_dummy_array(i_loc)
        else:
            loc_i = [i_loc]
        ## Same structure as input data
        if type(self.data_input) == np.ndarray:
            loc_i = np.array(loc_i)
        return loc_i

    def _get_loc_i_general_from_indices(self, i_loc):
        """Get element spatial information from spatial information.
        Format properly the input spatial information."""
        if type(i_loc) == list:
            if len(i_loc) == 0:
                return i_loc
            loc_i = self._get_loc_from_idxs(i_loc)
        elif type(i_loc) in [int, np.int32, np.int64]:
            loc_i = self._get_loc_from_idxs([i_loc])
        else:
            print i_loc, type(i_loc)
            raise TypeError("Not correct indice.")
        return loc_i

    def _get_loc_i_general(self, i_loc):
        """Get element spatial information. Generic function."""
        ## 0. Needed variable computations
        int_types = [int, np.int32, np.int64]
        ## 1. Loc retriever
        # If indice
        if type(i_loc) in int_types:
            loc_i = self._get_loc_from_idx(i_loc)
        # If List
        elif type(i_loc) == list:
            # Empty list
            if len(i_loc) == 0:
                return i_loc
            # if list of indices
            if type(i_loc[0]) in int_types:
                locs_i = []
                for i in i_loc:
                    loc_i = self._get_loc_from_idx(i)
                    locs_i.append(loc_i)
                ## Same structure as data
                if type(self.data_input) == np.ndarray:
                    locs_i = np.array(locs_i)
            # if list of objects data
            else:
                loc_i = i_loc
                ## Same structure as data
                if type(self.data_input) == np.ndarray:
                    loc_i = np.array(loc_i)
        # if coordinates
        elif type(i_loc) == np.ndarray:
            loc_i = self._get_loc_dummy_array(i_loc)
#        # if locations objects
#        elif isinstance(i_loc, Locations):
#            loc_i = self._get_loc_dummy_locations(i_loc)
        else:
            loc_i = self._get_loc_dummy(i_loc)
        return loc_i

    def _get_loc_dummy_array(self, i_loc):
        """Get location from coordinates array."""
#        sh = self.data_input.shape  # Global computation substitution
#        if len(np.array(i_loc).shape) == 1:
#            i_loc = np.array(i_loc).reshape((1, sh[1]))
        loc_i = i_loc
        return loc_i

#    def _get_loc_dummy_locations(self, i_loc):
#        """"""
#        loc_i = np.array(i_loc.locations).reshape((1, sh[1]))
#        return loc_i

#    def _get_loc_dummy(self, i_loc):
#        """Dummy get loc which return the exact input in array-like type."""
#        if type(i_loc) not in [np.ndarray, list]:
#            loc_i = [i_loc]
#        ## Same structure as data
#        if type(self.data_input) == np.ndarray:
#            loc_i = np.array(loc_i)
#        return i_loc

    ############################### Get indices ###############################
    ## Collapse to get_indice_i in _format_get_indice_i
    # Returns
    # -------
    # indices: list of ints
    #     the indices of the associeted spatial information elements input.
    #
    def _get_idx_from_loc_indata(self, loc_i, kr=0):
        """Get indices from stored data."""
        ### Check
        indices = []
        if type(self.data_input) == list:
            for j in range(len(self.data_input)):
                if self.data_input[j] == loc_i:
                    indices.append(j)
        else:
            aux_ind = np.where(self.data_input == loc_i)[0]
            if len(aux_ind):
                indices = list(aux_ind)
        return indices

    def _get_idxs_from_locs_notlistind(self, loc_i, kr=0):
        """Specific interaction with the data stored in retriever object."""
        data_idxs = []
        for i in range(len(loc_i)):
            data_idxs += self._get_idx_from_loc(loc_i[i], kr)
#        # WARNING: TODO: Probably outside when get_idx is used
#        if len(np.unique(data_idxs)) != len(data_idxs):
#            data_idxs = np.unique(data_idxs)
#        data_idxs = list(data_idxs)
        return data_idxs

    def _get_idxs_from_locs_listind(self, loc_i, kr=0):
        """Specific interaction with the data stored in retriever object."""
        i_locs = self._get_idx_from_loc(loc_i, kr)
        return i_locs

    def _get_indice_i_general_from_indices(self, i_loc, k=0, inorout=0):
        """Get indices of spatial information from spatial information.
        Format properly the input spatial information."""
        if type(i_loc) == list:
            loc_i = i_loc
        elif type(i_loc) in [int, np.int32, np.int64]:
            loc_i = [i_loc]
        else:
            raise TypeError("Not correct indice.")
        return loc_i

    def _get_indice_i_general_from_locations(self, loc_i, kr=0):
        """Get indices of spatial information from spatial information.
        Format properly the input spatial information."""
#        print '+'*20, loc_i, type(loc_i), self._get_idxs_from_locs
        if type(loc_i) == list:
            if len(loc_i) == 0:
                return loc_i
            if type(loc_i[0]) == np.ndarray:
                loc_i = np.array(loc_i)
            i_locs = self._get_idxs_from_locs(loc_i, kr)
        elif type(loc_i) == np.ndarray:
            i_locs = self._get_idxs_from_locs(loc_i, kr)
#            print '+'*5, i_locs, self._get_idxs_from_locs
        else:
            loc_i = [loc_i]
            i_locs = self._get_idxs_from_locs(loc_i, kr)
        return i_locs

    def _get_indice_i_general(self, loc_i, kr=0):
        """Get indice of spatial information. Generic function."""
        ## 0. Needed variable computations
        int_types = [int, np.int32, np.int64]
        # If indice
        if type(loc_i) in int_types:
            i_loc = self._get_idx_dummy(loc_i, kr)
        # If List
        elif type(loc_i) == list:
            # Empty list
            if len(loc_i) == 0:
                return loc_i
            # if list of indices
            if type(loc_i) in int_types:
                i_loc = self._get_idx_dummy(loc_i, kr)
            # if list of objects data
            else:
                i_loc = self._get_idxs_from_locs(loc_i, kr)
        # Objects or coordinates
        else:
            i_loc = self._get_idxs_from_locs(loc_i, kr)
        return i_loc

    def _get_idx_dummy(self, i_loc, kr=0):
        """Dummy index to index."""
        i_locs = i_loc if type(i_loc) == list else [i_loc]
        return i_locs

    ########################### Relativepos managing ##########################
    ## Collapse to _apply_relative_pos in _format_output_information
    # Returns
    # -------
    # neighs: list of np.ndarray or np.ndarray
    #     the neighs for each iss in i_loc
    # dists: list of list of np.ndarray or np.ndarray
    #     the information or relative position in respect to each iss in i_loc
    #
    def _general_relative_pos(self, neighs_info, element_i, element_neighs):
        """Intraclass interface for manage the interaction with relative
        position function."""
        if self.relative_pos is not None:
            ## Relative position computation
            if type(self.relative_pos).__name__ == 'function':
                rel_pos = self.relative_pos(element_i, element_neighs)
            else:
                rel_pos = self.relative_pos.compute(element_i, element_neighs)
            ## Neighs information
            if type(neighs_info) == tuple:
                neighs_info = neighs_info[0], rel_pos
            else:
                neighs_info = neighs_info, rel_pos
        return neighs_info

    def _dummy_relative_pos(self, neighs_info, element_i, element_neighs):
        """Not relative pos available."""
        return neighs_info

    def _apply_relative_pos_complete(self, res, point_i):
        loc_neighs = []
        for i in range(len(res[0])):
            loc_neighs_i = self._get_loc_from_idxs(res[0][i])
            loc_neighs.append(loc_neighs_i)
        res = self._apply_relative_pos(res, point_i, loc_neighs)
        return res

    def _apply_relative_pos_null(self, res, point_i):
        return res

    def _apply_preprocess_relative_pos_dim(self, res):
        for i in range(len(res)):
            res[i] = res[i].reshape((len(res[i]), 1))
        return res

    def _apply_preprocess_relative_pos_null(self, res):
        return res

    ###########################################################################
    ########################### Auxiliary functions ###########################
    ###########################################################################
    def __getitem__(self, i):
        "Perform the map assignation of the neighbourhood."
        neighs_info = self.retrieve_neighs(i)
        return neighs_info

    def __len__(self):
        return self._n0

    def export_neighs_info(self):
        return copy(self.neighs_info)

    @property
    def _n0(self):
        if self._heterogenous_input:
            raise Exception("Impossible action. Heterogenous input.")
        try:
            if self._autodata:
                n0 = len(self.retriever[0])
            else:
                n0 = len(self.data)
        except:
            n0 = len(self.data_input)
        return n0

    @property
    def _n1(self):
        if self._heterogenous_output:
            raise Exception("Impossible action. Heterogenous output.")
        try:
            n1 = np.prod(self.retriever[0].shape)
        except:
            n1 = len(self.data_output)
        return n1

    @property
    def shape(self):
        return (self._n0, self._n1)

    @property
    def data_input(self):
        if self._autodata:
            return self.retriever[0].data
        else:
            if self.data is None:
                self._autodata = True
                return self.data_input
            else:
                return self.data

    @property
    def data_output(self):
        return self.retriever[0].data

    def compute_neighnet(self, mapper, datavalue=None):
        """Compute the relations neighbours and build a network or multiplex
        with the defined retriever class.
        If we have an explicit retriever it is probably better to use algebra
        in order to vectorize computations.

        TODO
        ----
        * Check only 1dim rel_pos
        * Extend to k != 0
        * Accept a mapper if not heterogenous output

        Definition of heterogenous: len(output_map) == 1, same output for each retriever
        """
        ## 0. Conditions to ensure
        if self._heterogenous_output:
            msg = "Dangerous action. Heterogenous output."
            msg += "Only will be considered the 0 output_map"
            warnings.warn(msg)
        ## 00. Define global variables (TODO: Definition a priori)
#        n_data = self._ndim_rel_pos
#        neighs, dists = self[0]
#        try:
#            n_data = np.array(dists).shape[1]
#        except:
#            n_data = 1
        ks = self.neighs_info.ks
        ks = [0] if ks is None else ks
#        n_data = len(self.neighs_info.ks)
        sh = (self._n0, self._n1)
        ## 1. Computation
        # If explicit: (not the best way to use that, it is better to use algebra)
#        if self.type == 'explicit':
#            assert(type(mapper) in inttypes or mapper is None)
#            kr = 0 if mapper is None else mapper
#            return self.retrievers[kr].relations
        # else
        iss, jss = [[] for i in range(len(ks))], [[] for i in range(len(ks))]
        data = [[] for i in range(len(ks))]
        self.set_iter()
        for iss_i, neighs_info in self:
#        for i in xrange(sh[0]):
#            neighs_info = self[i]
            neighs, rel_pos, ks, iss_nei = neighs_info.get_information()
            ## Adding for each k perturbation if they are neighs to add
            for k in range(len(ks)):
                # Number of neighs for i in k perturbation
                n_i = len(neighs[k][0])
                if n_i != 0:
                    iss_i, jss_i = [i]*n_i, list(neighs[k][0])
                    iss[k].append(iss_i)
                    jss[k].append(jss_i)
                    ## Data definition
                    if datavalue is not None:
                        data[k] += [datavalue]*n_i
                    elif self._ifdistance is True:
                        data[k] += list(rel_pos[k][0])
                    else:
                        data[k] += [1.]*n_i
        ## 2. Format output
        # Concatenation
        iss = [np.hstack(iss[k]) for k in range(len(ks))]
        jss = [np.hstack(jss[k]) for k in range(len(ks))]
        data = [np.hstack(data[k]) for k in range(len(ks))]
        # Nets creations
        nets = []
        for k in range(len(ks)):
            nets.append(coo_matrix((data[k], (iss[k], jss[k])), shape=sh))
        if len(ks) == 1:
            nets = nets[0]
        return nets
