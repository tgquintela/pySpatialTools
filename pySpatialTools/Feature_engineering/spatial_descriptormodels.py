
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
from pySpatialTools.Retrieve import CollectionRetrievers
from aux_spatialdesc import Sp_DescriptorMapper


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
    ## Main classes
    descriptormodel = None
    retrievers = None
    ## Mapper
    _map_spdescriptor = 0
    ## Global params
    name_desc = None
    ## Parameters useful
    n_inputs = 0
    _pos_inputs = slice(0, 0, 1)

    def __init__(self, retrievers, descriptormodel, map_spdescriptor=None,
                 pos_inputs=None):
        self._format_retrievers(retrievers)
        self._format_descriptormdodels(descriptormodel)
        self._format_mapper(map_spdescriptor)
        self._format_loop(pos_inputs)

    def _format_retrievers(self, retrievers):
        "Formatter for retrievers."
        if type(retrievers) == list:
            self.retrievers = CollectionRetrievers(retrievers)
        else:
            self.retrievers = retrievers

    def _format_descriptormdodels(self, descriptormodel):
        self.descriptormodel = descriptormodel

    def _format_mapper(self, map_spdescriptor):
        "Format mapper."
        if map_spdescriptor is None:
            map_spdescriptor = Sp_DescriptorMapper()
        if type(map_spdescriptor) == np.ndarray:
            mapper = Sp_DescriptorMapper()
            mapper.set_from_array(map_spdescriptor)
            self._map_spdescriptor = mapper
        elif type(map_spdescriptor).__name__ == 'function':
            mapper = Sp_DescriptorMapper()
            mapper.set_from_function(map_spdescriptor)
            self._map_spdescriptor = mapper
        elif map_spdescriptor.__name__ == 'pst.Sp_DescriptorMapper':
            try:
                map_spdescriptor[0]
            except:
                msg = "Incorrect input for spatial descriptor mapper."
                raise TypeError(msg)
            self._map_spdescriptor = map_spdescriptor

    def _format_loop(self, pos_inputs):
        "Format the possible loop to go through."
        ## TODO: check coherence with retriever
        if pos_inputs is None:
            pos_inputs = self.retrievers.n_inputs
        if isinstance(pos_inputs, int):
            self.n_inputs = pos_inputs
        elif isinstance(pos_inputs, slice):
            n_inputs = (pos_inputs.stop+1-pos_inputs.start)/pos_inputs.step
            self.n_inputs = n_inputs
            self._pos_inputs = pos_inputs
        elif type(pos_inputs) not in [int, slice]:
            raise TypeError("Incorrect possible indices input.")

    def _get_methods(self, i):
        "Obtain the possible mappers we have to use in the process."
        methods = self._map_spdescriptor[i]
        staticneighs = methods[0]
        typeret = methods[1:3]
        typefeats = methods[3:]
        return staticneighs, typeret, typefeats

    def _get_methods_possibilies(self):
        pass

    ###########################################################################
    ###########################################################################
    def compute_descriptors(self, i):
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
        neighs_info = self.retrievers.retrieve_neighs(i, typeret_i=typeret)
        characs, vals_i =\
            self.descriptormodel.compute_descriptors(i, neighs_info,
                                                     typefeats=typefeats)
        return characs, vals_i

    def _compute_descriptors_seq1(self, i, typeret, typefeats):
        "Computation descriptors for aggregated data."
        k_rein = self.descriptormodel.features._k_reindices
        for k in range(k_rein):
            i_k = self.descriptormodel.features.apply_reindice(i, k)
            neighs_info =\
                self.retrievers.retrieve_neighs(i_k, typeret_i=typeret)
            characs, vals_i =\
                self.descriptormodel.compute_descriptors(i_k, neighs_info,
                                                         k, typefeats)
        return characs, vals_i

    ###########################################################################
    ###########################################################################
    def compute_nets(self):
        """Function used to compute the total measure.
        """
        desc = self.descriptormodel.initialization_output()
        for i in xrange(self.n_inputs):
            ## Compute descriptors for i
            desc_i, vals_i = self.compute_descriptors(i)
            ## Multiple mappings
            for k in range(len(desc_i)):
                desc[vals_i[k], :, k] =\
                    self.descriptormodel.add2result(desc[vals_i[k], :, k],
                                                    desc_i[k])
        desc = self.descriptormodel.to_complete_measure(desc)
        return desc

    def compute_nets_i(self):
        "Computation of the associate spatial descriptors for each i."
        for i in xrange(self.n_inputs):
            ## Compute descriptors for i
            desc_i, vals_i = self.compute_descriptors(i)
            yield desc_i, vals_i

    ###########################################################################
    ###########################################################################
    def compute_net_ik(self):
        """Function iterator used to get the result of each val_i and corr_i
        result for each combination of element i and permutation k.
        """
        for i in xrange(self.n_inputs):
            for k in range(self.reindices.shape[1]):
                # 1. Retrieve local characterizers
                desc_i, vals_i = self.compute_descriptors(i)
                for k in range(len(desc_i)):
                    yield vals_i[k], desc_i[k]

    def compute_process(self, logfile, lim_rows=0, n_procs=0,
                        proc_name="Descriptor model computation"):
        """Wrapper function to the spatialdescriptormodel process object. This
        processer contains tools of logging and storing information about the
        process.
        """
        modelproc = SpatialDescriptorModelProcess(self, logfile, lim_rows,
                                                  n_procs, proc_name)
        measure = modelproc.compute_measure()
        return measure
