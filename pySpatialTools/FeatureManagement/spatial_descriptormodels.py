
"""
Descriptor Model
----------------
Main class to the class of model descriptors. This class contains the main
functions and indications to compute the local descriptors.

TODO
----
- Online possibility (free input)

"""

import numpy as np
from process_descriptormodel import SpatialDescriptorModelProcess
from pySpatialTools.Retrieve import RetrieverManager
from pySpatialTools.utils.util_classes import Sp_DescriptorMapper
from ..utils import sp_general_filter_perturbations


class SpatialDescriptorModel:
    """The spatial descriptor model is an interface to compute spatial
    descriptors for a descriptor model processer mainly.
    It contains the utility classes of:
        * Retrievers: getting spatial neighbour elements.
        * Descriptor model: to transform it to spatial descriptors.
    Its main function in the process of computing descriptors from points is to
    manage the dealing with perturbation of the system for the sake of testing
    predictors and models.

    TODO
    ----
    - Return main parameters summary of the class
    - Run the process here
    -

    """

    def _initialization(self):
        ## Main classes
        self.retrievers = None
        self.featurers = None
        ## Mapper
        self._map_spdescriptor = 0
        ## Parameters useful
        self.n_inputs = 0
        self._pos_inputs = slice(0, 0, 1)
        self._map_indices = lambda self, i: i

    def __init__(self, retrievers, featurers, mapselector_spdescriptor=None,
                 pos_inputs=None, map_indices=None, perturbations=None,
                 aggregations=None, name_desc=None):
        self._initialization()
        self._format_retrievers(retrievers)
        self._format_featurers(featurers)
        self._format_perturbations(perturbations)
        self._format_mapper_selectors(mapselector_spdescriptor)
        self._format_loop(pos_inputs, map_indices)
        self._format_aggregations(aggregations)
        self._format_identifiers(name_desc)

    def compute(self, i=None):
        """Computation interface function."""
        if i is None:
            return self._compute_nets()
        else:
            return self._compute_descriptors(i)

    ################################ Formatters ###############################
    ###########################################################################
    def _format_retrievers(self, retrievers):
        "Formatter for retrievers."
        if type(retrievers) == list:
            self.retrievers = RetrieverManager(retrievers)
        else:
            self.retrievers = retrievers
        self.retrievers.set_neighs_info(True)

    def _format_perturbations(self, perturbations):
        """Format perturbations. TODO"""
        ## 0. Perturbations
        if perturbations is None:
            return
        ret_perturbs, feat_perturbs =\
            sp_general_filter_perturbations(perturbations)
        ## 1. Static neighbourhood (same neighs output for all k)
        aux = len(ret_perturbs) == 1 and ret_perturbs[0]._perturbtype == 'none'
        self._staticneighs = aux
        ## 2. Apply perturbations
        self.retrievers.add_perturbations(ret_perturbs)
        self.featurers.add_perturbations(feat_perturbs)

    def _format_aggregations(self, aggregations):
        """Prepare and add aggregations to retrievers and features."""
        if aggregations is None:
            return
        if type(aggregations) == list:
            for i in range(len(aggregations)):
                self._format_aggregations(aggregations[i])
        if type(aggregations) == tuple:
            # Add aggregations to retrievers
            self.retrievers.add_aggregations(*aggregations)
            # Add aggregations to features
            self.featurers.add_aggregations(*aggregations)
        else:
            # Add aggregations to retrievers
            self.retrievers.add_aggregations(aggregations)
            # Add aggregations to features
            self.featurers.add_aggregations(*aggregations)

    def _format_featurers(self, featurers):
        self.featurers = featurers

    def _format_mapper_selectors(self, _mapselector_spdescriptor):
        "Format selectors."
        if _mapselector_spdescriptor is None:
            _mapselector_spdescriptor = Sp_DescriptorMapper()
        if type(_mapselector_spdescriptor) == np.ndarray:
            mapperselector = Sp_DescriptorMapper()
            mapperselector.set_from_array(_mapselector_spdescriptor)
            self._mapselector_spdescriptor = mapperselector
        elif type(_mapselector_spdescriptor).__name__ == 'function':
            mapperselector = Sp_DescriptorMapper()
            mapperselector.set_from_function(_mapselector_spdescriptor)
            self._mapselector_spdescriptor = mapperselector
        elif _mapselector_spdescriptor.__name__ == 'pst.Sp_DescriptorMapper':
            try:
                _mapselector_spdescriptor[0]
            except:
                msg = "Incorrect input for spatial descriptor mapperselector."
                raise TypeError(msg)
            self._mapselector_spdescriptor = _mapselector_spdescriptor
        ##### TEMPORAL
        self._mapselector_spdescriptor = lambda idx: (0, 0, 0, 0, 0, 0, 0)

    def _format_loop(self, pos_inputs, map_indices):
        "Format the possible loop to go through."
        ## TODO: check coherence with retriever
        if pos_inputs is None:
            pos_inputs = self.retrievers.n_inputs
        if isinstance(pos_inputs, int):
            self.n_inputs = pos_inputs
            self._pos_inputs = slice(0, pos_inputs, 1)
        elif isinstance(pos_inputs, tuple):
            step = 1 if len(pos_inputs) == 2 else pos_inputs[2]
            self.n_inputs = pos_inputs[1]-pos_inputs[0]
            self._pos_inputs = slice(pos_inputs[0], pos_inputs[1], step)
        elif isinstance(pos_inputs, slice):
            st0, st1, stp = pos_inputs.start, pos_inputs.stop, pos_inputs.step
            n_inputs = len(range(st0, st1, stp))
            self.n_inputs = n_inputs
            self._pos_inputs = pos_inputs
        elif type(pos_inputs) not in [int, slice]:
            raise TypeError("Incorrect possible indices input.")
        ## Create map_indices
        if map_indices is None:
            def map_indices(s, i):
                if s._pos_inputs is not None:
                    return s._pos_inputs.start + s._pos_inputs.step*i
                else:
                    return i
        self._map_indices = map_indices

        ## Create iterator
        def iter_indices(self):
            start, stop = self._pos_inputs.start, self._pos_inputs.stop
            step = self._pos_inputs.step
            for idx in xrange(start, stop, step):
                yield idx
        self.iter_indices = iter_indices
        ## Notice to featurer
        self.featurers.set_map_vals_i(pos_inputs)

    def _format_identifiers(self, name_desc):
        """Format information of the method applied."""
        if name_desc is None or type(name_desc) != str:
            self.name_desc = self.featurers.descriptormodel.name_desc
        else:
            self.name_desc = name_desc

    ################################# Getters #################################
    ###########################################################################
    def _get_methods(self, i):
        "Obtain the possible mappers we have to use in the process."
        methods = self._mapselector_spdescriptor(i)
        staticneighs = self.retrievers.staticneighs
        typeret = methods[:2]
        typefeats = methods[2:]
        return staticneighs, typeret, typefeats

    ################################# Setters #################################
    ###########################################################################
    def add_perturbations(self, perturbations):
        """Add perturbations to the spatial descriptormodel."""
        self._format_perturbations(perturbations)

    def add_aggregations(self, aggregations):
        """Add aggregations to the spatial descriptor model."""
        self._format_aggregations(aggregations)

    def set_loop(self, pos_inputs, map_indices=None):
        """Set loop in order to get only reduced possibilities."""
        self._format_loop(pos_inputs, map_indices)

    ############################ Computer functions ###########################
    ###########################################################################
    def _compute_nets(self):
        """Function used to compute the total measure.
        """
        desc = self.featurers.initialization_output()
        print 'x'*20, desc
        for i in self.iter_indices(self):
            ## Compute descriptors for i
            desc_i, vals_i = self._compute_descriptors(i)
            print 'y'*25, desc_i, vals_i
            desc = self.featurers.add2result(desc, desc_i, vals_i)
        print desc
        desc = self.featurers.to_complete_measure(desc)
        return desc

    def _compute_retdriven(self):
        desc = self.featurers.initialization_output()
        k_pert = self.featurers.k_perturb+1
        ks = list(range(k_pert))
        _, typeret, typefeats = self._get_methods(i)
        self.retrievers.set_typeret(typeret)
        for iss, neighs_info in self.retrievers:
            characs_iss, vals_iss =\
                self.featurers.compute_descriptors(iss, neighs_info,
                                                   ks, typefeats)
            desc = self.featurers.add2result(desc, characs_iss, vals_iss)
        desc = self.featurers.to_complete_measure(desc)
        return desc

    def _compute_descriptors(self, i):
        "Compute the descriptors assigned to element i."
        staticneighs, typeret, typefeats = self._get_methods(i)
        if staticneighs:
            characs, vals_i = self._compute_descriptors_seq0(i, typeret,
                                                             typefeats)
        else:
            characs, vals_i = self._compute_descriptors_seq1(i, typeret,
                                                             typefeats)

        return characs, vals_i

    def _compute_descriptors_seq0(self, i, typeret, typefeats):
        "Computation descriptors for non-aggregated data."
        ## Model1
        staticneighs, _, _ = self._get_methods(i)
        k_pert = self.featurers.k_perturb+1
        ks = list(range(k_pert))
        neighs_info =\
            self.retrievers.retrieve_neighs(i, typeret_i=typeret, k=ks)
        print staticneighs, neighs_info.staticneighs
        assert(staticneighs == neighs_info.staticneighs)
        characs, vals_i =\
            self.featurers.compute_descriptors(i, neighs_info, ks, typefeats)
        return characs, vals_i

    def _compute_descriptors_seq1(self, i, typeret, typefeats):
        "Computation descriptors for aggregated data."
        k_pert = self.featurers.k_perturb+1
        characs, vals_i = [], []
        for k in range(k_pert):
            neighs_info =\
                self.retrievers.retrieve_neighs(i, typeret_i=typeret, k=k)
            assert(len(neighs_info.ks) == 1)
            assert(neighs_info.ks[0] == k)
            characs_k, vals_i_k =\
                self.featurers.compute_descriptors(i, neighs_info,
                                                   k, typefeats)
            characs.append(characs_k)
            vals_i.append(vals_i_k)
        ## Joining descriptors from different perturbations
        characs = self.featurers._join_descriptors(characs)
        vals_i = np.concatenate(vals_i)
        return characs, vals_i

    ################################ ITERATORS ################################
    ###########################################################################
    def compute_nets_i(self):
        """Computation of the associate spatial descriptors for each i."""
        for i in self.iter_indices(self):
            ## Compute descriptors for i
            desc_i, vals_i = self._compute_descriptors(i)
            yield desc_i, vals_i

    def compute_net_ik(self):
        """Function iterator used to get the result of each val_i and corr_i
        result for each combination of element i and permutation k.
        """
        for i in self.iter_indices(self):
            for k in range(self.reindices.shape[1]):
                # 1. Retrieve local characterizers
                desc_i, vals_i = self._compute_descriptors(i)
                for k in range(len(desc_i)):
                    yield vals_i[k], desc_i[k]

    ############################ Process function #############################
    ###########################################################################
    def compute_process(self, logfile, lim_rows=0, n_procs=0):
        """Wrapper function to the spatialdescriptormodel process object. This
        processer contains tools of logging and storing information about the
        process.
        """
        modelproc = SpatialDescriptorModelProcess(self, logfile, lim_rows,
                                                  n_procs)
        measure = modelproc.compute_measure()
        return measure
