
"""
Retrievers
----------
The objects to retrieve neighbours in flat (non-splitted) space or precomputed
mapped relations.


Structure:
----------


TODO:
----
- Ifdistance better implementation
- Exclude better implementation
- Multiple regions
- Multiple points to get neighs

"""

import numpy as np
import warnings
from scipy.sparse import coo_matrix
from aux_retriever import _check_retriever
from ..utils import NonePerturbation
from ..utils import ret_filter_perturbations
from ..utils.util_classes import SpatialElementsCollection, Locations


class Retriever:
    """Class which contains the retriever of elements.
    """
    __name__ = 'pySpatialTools.Retriever'

    def _retrieve_neighs_static(self, i_loc):
        """Retrieve neighs and distances. This function acts as a wrapper to
        more specific functions designed in the specific classes and methods.
        This function is composed by mutable functions in order to take profit
        of saving times excluding flags. It assumes staticneighs.
        """
        ## 1. Retrieve neighs
        neighs, dists = self._retrieve_neighs_spec(i_loc, {})
        ## 2. Format output
        neighs_info = self._format_output(i_loc, neighs, dists)
        return neighs_info

    def _retrieve_neighs_dynamic(self, i_loc):
        """Retrieve neighs and distances. This function acts as a wrapper to
        more specific functions designed in the specific classes and methods.
        This function is composed by mutable functions in order to take profit
        of saving times excluding flags. It assumes different preset
        perturbations to retrieve.
        """
        neighs_info = []
        for k in range(self.k_perturb+1):
            ## 1. Map perturb
            _, k_r = self._map_perturb(k)
            ## 2. Retrieve neighs
            neighs, dists = self._retrieve_neighs_spec(i_loc, {}, k_r=k_r)
            nei_k = self._format_output(i_loc, neighs, dists, k_r=k_r)
            neighs_info.append(nei_k)
        return neighs_info

    def _retrieve_neighs_general(self, i_loc, info_i={}, ifdistance=None,
                                 k=None, output=0):
        """Retrieve neighs and distances. This function acts as a wrapper to
        more specific functions designed in the specific classes and methods.
        """
        ## 0. Prepare variables
        info_i, ifdistance, ks =\
            self._format_inputs_retriever(i_loc, info_i, ifdistance, k, output)
        ## 1. Retrieve neighs
        if ks == 0 or self.staticneighs:
            # Get neighs info
            neighs, dists =\
                self._retrieve_neighs_spec(i_loc, info_i, ifdistance)
            ## 2. Format output
            neighs_info = self._format_output(i_loc, neighs, dists, output)
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
        return neighs_info

    def _format_inputs_retriever(self, i_loc, info_i, ifdistance, k, output):
        """Format inputs retriever check and format the inputs for retrieving.
        """
        # Prepare information retrieve
        info_i = self._get_info_i(i_loc, info_i)
        #i_loc = self._get_loc_i(i_loc)
        ifdistance = self._ifdistance if ifdistance is None else ifdistance
        # Prepare perturbation index
        ks = [k] if type(k) == int else k
        ks = 0 if ks is None else ks
        ## Check output (TODO)
        return info_i, ifdistance, ks

    ######################### Perturbation management #########################
    ###########################################################################
    def add_perturbations(self, perturbations):
        """Add perturbations."""
        perturbations = ret_filter_perturbations(perturbations)
        if type(perturbations) == list:
            for p in perturbations:
                self._dim_perturb.append(p.k_perturb)
                self._perturbators.append(p)
                self._create_map_perturbation()
                self._add_perturbated_retrievers(p)
        else:
            self._dim_perturb.append(perturbations.k_perturb)
            self._perturbators.append(perturbations)
            self._create_map_perturbation()
            self._add_perturbated_retrievers(perturbations)

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
            for k in range(perturbation.k_perturb):
                locs_p = perturbation.apply2locs(self.retriever[0].data, k=k)
                self._define_retriever(locs_p[:, :, 0])

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
        self._autodata = False
        self._virtual_data = False
        ## Retriever information
        self.retriever = []
        self._info_ret = None
        self._info_f = None
        ## Perturbation
        self._dim_perturb = [1]
        self._map_perturb = lambda x: (0, 0)
        self._perturbators = [NonePerturbation()]
        self.staticneighs = True
        ## External objects to apply
        self.relative_pos = None
        ## IO information
        self._autoexclude = False
        self._ifdistance = False
        self._autoret = False
        self._heterogenous_input = False
        self._heterogenous_output = False
        ## IO methods
        self._input_map = lambda s, i: i
        self._output_map = [lambda s, i, x: x]
        ## Check
        _check_retriever(self)

    ################################ Formatters ###############################
    ###########################################################################
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
        if ifdistance is None:
            pass
        else:
            if ifdistance:
                pass
            else:
                pass
        ## Relative position managing
        self.relative_pos = relativepos
        if relativepos is None:
            self._apply_relative_pos = self._dummy_relative_pos
        else:
            self._apply_relative_pos = self._general_relative_pos

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
            if info_f is None and info_ret is None:
                self._get_info_i = self._dummy_get_info_i
            elif info_f is None:
                ## TODO: Another flag for the case of indexable get_info_i
                self._get_info_i = self._dummy_get_info_i_stored
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
            if self._ifdistance:
                self._retrieve_neighs_spec =\
                    self._retrieve_neighs_constant_distance
            else:
                self._retrieve_neighs_spec =\
                    self._retrieve_neighs_constant_nodistance
            if self.k_perturb == 0:
                self.retrieve_neighs = self._retrieve_neighs_static
            else:
                self.retrieve_neighs = self._retrieve_neighs_dynamic
        ## Format specific function
        else:
            self.retrieve_neighs = self._retrieve_neighs_general
            self._retrieve_neighs_spec = self._retrieve_neighs_general_spec
        ## Format retrieve locs and indices
        self._format_get_loc_i()
        self._format_get_indice_i()

    def _format_preparators(self, bool_input_idx):
        """Format the prepare inputs function in order to be used properly and
        efficient avoiding extra useless computations.
        """
        if self.preferable_input_idx == bool_input_idx:
            self._prepare_input = self._dummy_prepare_input
        elif self.preferable_input_idx:
            self._prepare_input = self._dummy_loc2idx_prepare_input
        elif not self.preferable_input_idx:
            self._prepare_input = self._dummy_idx2loc_prepare_input

    ################################# Auxiliar ################################
    ###########################################################################
    def _exclude_auto(self, i_loc, neighs, dists, kr=0):
        """Exclude auto elements if there exist in the neighs retrieved.
        This is a generic function independent on the type of the element.
        """
        ## 0. Detect input i_loc and retrieve to_exclude_elements list
        # If it is an indice
        if type(i_loc) in [int, np.int32, np.int64]:
            to_exclude_elements = [i_loc]
        # If it is an element spatial information
        else:
            to_exclude_elements = self._build_excluded_elements(i_loc, kr)
        ## 1. Excluding task
        n_p = np.array(neighs).shape[0]
        idxs_exclude = [i for i in xrange(n_p) if neighs[i]
                        in to_exclude_elements]
        neighs = [neighs[i] for i in xrange(n_p) if i not in idxs_exclude]
        if dists is not None:
            dists = [dists[i] for i in xrange(n_p) if i not in idxs_exclude]
        return neighs, dists

    def _build_excluded_elements(self, i_loc, kr=0):
        """Build the excluded points from i_loc if it is not an index."""
        if type(i_loc) == np.ndarray:
            sh = i_loc.shape
            i_loc = i_loc if len(sh) == 2 else i_loc.reshape(1, sh[0])
        try:
            logi = np.all(self.retriever[kr].data == i_loc, axis=1).ravel()
        except:
            try:
                logi = np.all(self.retriever[kr].data == i_loc)
            except:
                n = len(self.retriever[kr].data)
                logi = np.array([self.retriever[kr].data[i] == i_loc
                                 for i in xrange(n)])
        assert len(logi) == len(self.retriever[kr].data)
        to_exclude_points = np.where(logi)[0]
        return to_exclude_points

    ############################# InfoRet managing ############################
    ###########################################################################
    ## Collapse to _get_info_i in _format_retriever_info
    def _dummy_get_info_i(self, i_loc, info_i):
        return info_i

    def _dummy_get_info_i_stored(self, i_loc, info_i=None):
        """Dummy get retrieve information."""
        return self._info_ret

    def _dummy_get_info_i_indexed(self, i_loc, info_i=None):
        """Dummy indexable retrieve information."""
        return self._info_ret[i_loc]

    def _dummy_get_info_i_f(self, i_loc, info_i=None):
        """Dummy get retrieve information using function."""
        return self._info_f(i_loc, self._info_ret)

    def _general_get_info_i(self, i_loc, info_i):
        """Get retrieving information for each element i_loc. Comunicate the
        input i with the data_input. It is a generic function independent on
        the type of the elements we want to retrieve.
        """
        if not info_i:
            if type(i_loc) in [int, np.int32, np.int64]:
                if type(self._info_ret) in [list, np.ndarray]:
                    info_i = self._info_ret[i_loc]
                else:
                    info_i = self._info_ret
            else:
                if self._info_f is None:
                    return {}
                if type(self._info_f).__name__ == 'name':
                    info_i = self._info_f(i_loc, info_i)
                else:
                    raise TypeError("self._info_f not defined properly.")
        return info_i

    ########################### GetLocation managing ##########################
    ###########################################################################
    ## Collapse to _prepare_input in _format_preparators
    def _dummy_prepare_input(self, i_loc, kr=0):
        """Dummy prepare input."""
        i_mloc = self._input_map(self, i_loc)
        return i_mloc

    def _dummy_idx2loc_prepare_input(self, i_loc, kr=0):
        """Dummy prepare input transforming indice to location."""
        i_loc = self._input_map(self, i_loc)
        loc_i = self.get_loc_i(i_loc, kr)
        return loc_i

    def _dummy_loc2idx_prepare_input(self, loc_i, kr=0):
        """Dummy prepare input transforming location to indice."""
        loc_i = self._input_map(self, loc_i)
        i_loc = self.get_indice_i(loc_i, kr)
        return i_loc

    def _format_get_loc_i(self):
        """Format the get indice function."""
        if self._constant_ret:
            try:
                self._get_loc_from_idx_virtual(0, 0)
                self.get_loc_i = self._get_loc_from_idx_virtual
            except:
                self.get_loc_i = self._get_loc_from_idx
        else:
            self.get_loc_i = self._get_loc_i_general

    def _get_loc_from_idx_virtual(self, i_loc, kr=0):
        """Get location from indice in virtual data retriever."""
        loc = self.retriever[kr].get_location(i_loc)
        return loc

    def _get_loc_from_idx(self, i_loc, kr=0):
        """Get location from indice in explicit data retriever."""
        loc = np.array(self.retriever[kr].data[i_loc])
        return loc

    def _get_loc_i_general(self, i_loc, k=0, inorout=True):
        """Get element spatial information. Generic function."""
        ## 0. Needed variable computations
        ifdata = inorout and not self._autodata
        sh = self.data_input.shape
        if ifdata:
            flag = isinstance(self.data, Locations)
        else:
            flag = isinstance(self.retriever[k].data, Locations)
        ## 1. Loc retriever
        if type(i_loc) in [int, np.int32, np.int64]:
            try:
                if flag:
                    if ifdata:
                        i_loc = self.data_input[i_loc].location
                        i_loc = np.array(i_loc).reshape((1, sh[1]))
                    else:
                        i_loc = self.data_input[i_loc].location
                        i_loc = np.array(i_loc).reshape((1, sh[1]))
                else:
                    if ifdata:
                        loc_i = np.array(self.data[i_loc]).reshape((1, sh[1]))
                    else:
                        loc_i = np.array(self.retriever[k].data[i_loc])
                        loc_i = loc_i.reshape((1, sh[1]))
            except:
                if ifdata:
                    loc_i = self.data_input[i_loc]
        elif type(i_loc) in [list, np.ndarray]:
            loc_i = np.array(i_loc).reshape((1, sh[1]))
        elif isinstance(i_loc, Locations):
            i_loc = np.array(i_loc.locations).reshape((1, sh[1]))
        else:
            loc_i = i_loc
        return loc_i

    def _format_get_indice_i(self):
        """Format the get indice function."""
        if self._constant_ret:
            try:
                loc = self._get_loc_from_idx_virtual(0, 0)
                self._get_indice_i_virtual(loc, 0)
                self.get_indice_i = self._get_indice_i_virtual
            except:
                loc = self._get_loc_from_idx(0, 0)
                try:
                    self._get_indice_i_global(loc, 0)
                    self.get_indice_i = self._get_indice_i_global
                except:
                    self._get_indice_i_elementwise(loc, 0)
                    self.get_indice_i = self._get_indice_i_elementwise
        else:
            self.get_indice_i = self._get_indice_i_general

    def _get_indice_i_general(self, loc_i, kr=0):
        """Obtain the indices from the elements, element-wise."""
        try:
            indice = self._get_indice_i_virtual(loc_i, kr)
        except:
            try:
                indice = self._get_indice_i_global(loc_i, kr)
            except:
                indice = self._get_indice_i_elementwise(loc_i, kr)
        return indice

    def _get_indice_i_virtual(self, loc_i, kr=0):
        """Get indice for virtual (not computed explicitely) data."""
        indice = self.retriever[kr].get_indice(loc_i)
        return indice

    def _get_indice_i_global(self, loc_i, kr=0):
        """Global search from elements."""
        indice = np.where(self.retriever[kr].data == loc_i)[0]
        return indice

    def _get_indice_i_elementwise(self, loc_i, kr=0):
        """Obtain the indices from the elements element-wise."""
        indice = np.where([self.retriever[kr].data[i] == loc_i
                           for i in range(len(self.retriever[kr].data))])
        return indice

    ########################### Relativepos managing ##########################
    ###########################################################################
    ## Collapse to _apply_relative_pos in _format_output_information
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

    ###########################################################################
    ########################### Auxiliary functions ###########################
    ###########################################################################
    def __getitem__(self, i):
        "Perform the map assignation of the neighbourhood."
        neighs, dists = self.retrieve_neighs(i)
        return neighs, dists

    def __len__(self):
        return self._n0

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
            return np.array(self.retriever[0].data)
        else:
            if self.data is None:
                self._autodata = True
                return np.array(self.data_input)
            else:
                return np.array(self.data)

    @property
    def data_output(self):
        return np.array(self.retriever[0].data)

    def compute_neighnet(self):
        """Compute the relations neighbours and build a network or multiplex
        with the defined retriever class"""
        ## 0. Conditions to ensure
        if self._heterogenous_output:
            msg = "Dangerous action. Heterogenous output."
            msg += "Only will be considered the 0 output_map"
            warnings.warn(msg)
        ## 00. Define global variables
        neighs, dists = self[0]
        try:
            n_data = np.array(dists).shape[1]
        except:
            n_data = 1
        sh = (self._n0, self._n1)
        ## 1. Computation
        iss, jss = [], []
        data = [[] for i in range(n_data)]
        for i in xrange(self._n0):
            neighs, dists = self[i]
            #dists = np.array(dists).reshape((len(dists), n_data))
            n_i = len(neighs)
            if n_i != 0:
                iss_i, jss_i = [i]*n_i, list(neighs)
                iss.append(iss_i)
                jss.append(jss_i)
                for k in range(n_data):
                    data[k] += dists
        ## 2. Format output
        iss, jss = np.hstack(iss), np.hstack(jss)
        data = [np.hstack(data[k]) for k in range(n_data)]
        nets = []
        for k in range(n_data):
            nets.append(coo_matrix((data[k], (iss, jss)), shape=sh))
        if n_data == 1:
            nets = nets[0]
        return nets
