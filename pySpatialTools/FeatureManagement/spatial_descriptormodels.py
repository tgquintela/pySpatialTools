
"""
Descriptor Model
----------------
Main class to the class of model descriptors. This class contains the main
functions and indications to compute the local descriptors.

TODO
----
- Online possibility (free input)

"""

## General imports
import numpy as np
## General objects
from process_descriptormodel import SpatialDescriptorModelProcess
from pySpatialTools.Retrieve import RetrieverManager
from pySpatialTools.utils.selectors import Sp_DescriptorSelector
from features_retriever import FeaturesManager
## Special tools functions
from pySpatialTools.Discretization import _discretization_parsing_creation,\
    _discretization_information_creation
from pySpatialTools.Retrieve.tools_retriever import create_aggretriever
from features_objects import _featuresobject_parsing_creation
from pySpatialTools.utils.perturbations import sp_general_filter_perturbations


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
        self.selectors = None
#        self._default_selectors = (0, 0), (0, 0, 0, 0, 0, 0)
        self._default_selectors = None, None
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
        elif isinstance(retrievers, RetrieverManager):
            self.retrievers = retrievers
        else:
            self.retrievers = RetrieverManager(retrievers)
        self.retrievers.set_neighs_info(True)

    def _format_perturbations(self, perturbations):
        """Format perturbations. TODO"""
        ## 0. Perturbations processing
        if perturbations is None:
            return
        ret_perturbs, feat_perturbs =\
            sp_general_filter_perturbations(perturbations)
#        ## 1. Static neighbourhood (same neighs output for all k)
#        aux = len(ret_perturbs) == 1 and ret_perturbs[0]._perturbtype == 'none'
#        self._staticneighs = aux
        ## 1. Apply perturbations
        self.retrievers.add_perturbations(ret_perturbs)
        self.featurers.add_perturbations(feat_perturbs)
        assert(self.retrievers.k_perturb == self.featurers.k_perturb)

    def _format_aggregations(self, aggregations, i_r=(None, None)):
        """Prepare and add aggregations to retrievers and features."""
        if aggregations is None:
            return
        if type(aggregations) == list:
            for i in range(len(aggregations)):
                self._format_aggregations(aggregations[i], i_r)
        if type(aggregations) == tuple:
            ## Prepare instructions
            i_ret = i_r[0]
            i_ret = range(len(self.retrievers)) if i_ret is None else i_ret
            i_ret = [i_ret] if type(i_ret) != list else i_ret
            i_feat = i_r[1]
            i_feat = range(len(self.featurers)) if i_feat is None else i_feat
            i_feat = [i_feat]*len(i_ret) if type(i_feat) != list else i_feat
            ## Assert correctness
            assert(len(i_ret) == len(i_feat))
            ## Main loop
            for i in range(len(i_ret)):
                ## Preparing information to retriever number i_ret
                ret = self.retrievers.retrievers[i_ret[i]]
                agg_0 = _discretization_information_creation(aggregations[0],
                                                             ret)
                aggregations_i = tuple([agg_0] + list(aggregations[1:]))
                # Add aggregation to retrievers
                new_ret = create_aggretriever(aggregations_i)
                self.retrievers.add_aggregations(new_ret)
                # Add aggregations to features
                i_feat_i = [i_feat[i]] if type(i_feat[i]) == int else i_feat[i]
                for j in i_feat_i:
                    new_features =\
                        create_aggfeatures(aggregations_i,
                                           self.featurers.features[j])
                    self.featurers.add_aggregations(new_features)

    def _format_featurers(self, featurers):
        if isinstance(featurers, FeaturesManager):
            self.featurers = featurers
        else:
            self.featurers = FeaturesManager(featurers)

    def _format_mapper_selectors(self, _mapselector_spdescriptor):
        "Format selectors."
        self.selectors = self._default_selectors
        if _mapselector_spdescriptor is None:
            self._mapselector_spdescriptor =\
                self._mapselector_spdescriptor_null
        if type(_mapselector_spdescriptor) == np.ndarray:
            assert(len(_mapselector_spdescriptor.shape) == 2)
            assert(_mapselector_spdescriptor.shape[1] == 8)
            sels = (_mapselector_spdescriptor[:, 0:2].astype(int),
                    [_mapselector_spdescriptor[:, 2:4].astype(int),
                     _mapselector_spdescriptor[:, 4:6].astype(int),
                     _mapselector_spdescriptor[:, 6:8].astype(int)])
            self.retrievers.set_selector(sels[0])
            self.featurers.set_selector(*sels[1])
            self._mapselector_spdescriptor =\
                self._mapselector_spdescriptor_null
        elif type(_mapselector_spdescriptor) == tuple:
            if type(_mapselector_spdescriptor[0]) == int:
                assert(len(_mapselector_spdescriptor) == 8)
                sels = (_mapselector_spdescriptor[:2],
                        [_mapselector_spdescriptor[2:4],
                         _mapselector_spdescriptor[4:6],
                         _mapselector_spdescriptor[6:8]])
                self.retrievers.set_selector(sels[0])
                self.featurers.set_selector(*sels[1])
                self._mapselector_spdescriptor =\
                    self._mapselector_spdescriptor_null
            elif type(_mapselector_spdescriptor[0]) == tuple:
                assert(len(_mapselector_spdescriptor) == 2)
                assert(len(_mapselector_spdescriptor[0]) == 2)
                if len(_mapselector_spdescriptor[1]) == 6:
                    sels = (_mapselector_spdescriptor[0],
                            [_mapselector_spdescriptor[1][:2],
                             _mapselector_spdescriptor[1][2:4],
                             _mapselector_spdescriptor[1][4:]])
                else:
                    assert(len(_mapselector_spdescriptor[1]) == 3)
                    logi = [len(e) == 2 for e in _mapselector_spdescriptor[1]]
                    assert(all(logi))
                    sels = _mapselector_spdescriptor
                self.retrievers.set_selector(sels[0])
                self.featurers.set_selector(*sels[1])
                self._mapselector_spdescriptor =\
                    self._mapselector_spdescriptor_null
            elif type(_mapselector_spdescriptor[0]) == np.ndarray:
                assert(len(_mapselector_spdescriptor) == 2)
                assert(len(_mapselector_spdescriptor[0].shape) == 2)
                assert(_mapselector_spdescriptor[0].shape[1] == 2)
                if type(_mapselector_spdescriptor[1]) == tuple:
                    logi = [e.shape[1] == 2
                            for e in _mapselector_spdescriptor[1]]
                    assert(all(logi))
                    sels = _mapselector_spdescriptor
                else:
                    assert(_mapselector_spdescriptor[1].shape[1] == 6)
                    assert(len(_mapselector_spdescriptor[1].shape) == 2)
                    sels = (_mapselector_spdescriptor[0].astype(int),
                            [_mapselector_spdescriptor[1][:, :2].astype(int),
                             _mapselector_spdescriptor[1][:, 2:4].astype(int),
                             _mapselector_spdescriptor[1][:, 4:].astype(int)])
                self.retrievers.set_selector(sels[0])
                self.featurers.set_selector(*sels[1])
                self._mapselector_spdescriptor =\
                    self._mapselector_spdescriptor_null
            elif type(_mapselector_spdescriptor[0]).__name__ == 'function':
                assert(len(_mapselector_spdescriptor) == 2)
                self.retrievers.set_selector(_mapselector_spdescriptor[0])
                self.featurers.set_selector(_mapselector_spdescriptor[1])
                self._mapselector_spdescriptor =\
                    self._mapselector_spdescriptor_null
        elif type(_mapselector_spdescriptor).__name__ == 'function':
            self.selectors = Sp_DescriptorSelector(_mapselector_spdescriptor)
#            mapperselector.set_from_function(_mapselector_spdescriptor)
            self._mapselector_spdescriptor =\
                self._mapselector_spdescriptor_selector
        elif isinstance(_mapselector_spdescriptor, Sp_DescriptorSelector):
            self.selectors = _mapselector_spdescriptor
            self._mapselector_spdescriptor =\
                self._mapselector_spdescriptor_selector
#            try:
#                _mapselector_spdescriptor[0]
#            except:
#                msg = "Incorrect input for spatial descriptor mapperselector."
#                raise TypeError(msg)

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
                return s._pos_inputs.start + s._pos_inputs.step*i
#                if s._pos_inputs is not None:
#                    return s._pos_inputs.start + s._pos_inputs.step*i
#                else:
#                    return i
        self._map_indices = map_indices
        ## Notice to featurer
        self.featurers.set_map_vals_i(pos_inputs)

    def _format_identifiers(self, name_desc):
        """Format information of the method applied."""
        if name_desc is None or type(name_desc) != str:
            self.name_desc = self.featurers.descriptormodels[0].name_desc
        else:
            self.name_desc = name_desc

    ################################# Getters #################################
    ###########################################################################
    def _get_methods(self, i):
        "Obtain the possible mappers we have to use in the process."
        staticneighs = self.retrievers.staticneighs
        methods = self._mapselector_spdescriptor(i)
        if type(methods) == list:
            typeret, typefeats = [], []
            for e in methods:
                e1, e2 = e
                typeret.append(e1)
                typefeats.append(e2)
        else:
            typeret, typefeats = methods
        return staticneighs, typeret, typefeats

    def _mapselector_spdescriptor_null(self, i):
        return self._default_selectors

#    def _mapselector_spdescriptor_constant(self, i):
#        i_len = 1 if type(i) == int else len(i)
#        logi = type(i) == int
#        if logi:
#            return self.selectors
#        else:
#            return [self.selectors[0]]*i_len, [self.selectors[1]]*i_len

    def _mapselector_spdescriptor_selector(self, i):
        return self.selectors[i]

    def iter_indices(self):
        """Get indices in iteration of indices."""
        start, stop = self._pos_inputs.start, self._pos_inputs.stop
        step = self._pos_inputs.step
        for idx in xrange(start, stop, step):
            yield idx

    ################################# Setters #################################
    ###########################################################################
    def add_perturbations(self, perturbations):
        """Add perturbations to the spatial descriptormodel."""
        self._format_perturbations(perturbations)

    def add_aggregations(self, aggregations, i_r=(None, None)):
        """Add aggregations to the spatial descriptor model."""
        self._format_aggregations(aggregations, i_r)

    def set_loop(self, pos_inputs, map_indices=None):
        """Set loop in order to get only reduced possibilities."""
        self._format_loop(pos_inputs, map_indices)

    ############################ Computer functions ###########################
    ###########################################################################
    def _compute_nets(self):
        """Function used to compute the total measure.
        """
        desc = self.featurers.initialization_output()
#        print 'x'*20, desc
        for i in self.iter_indices():
            ## Compute descriptors for i
            desc_i, vals_i = self._compute_descriptors(i)
#            print 'y'*25, desc_i, vals_i
            desc = self.featurers.add2result(desc, desc_i, vals_i)
#        print desc
        desc = self.featurers.to_complete_measure(desc)
        return desc

    def _compute_retdriven(self):
        """Compute the whole spatial descriptor measure let the retrievers
        drive the process."""
#        _, typeret, typefeats = self._get_methods(i)
#        self.retrievers.set_typeret(typeret)
        desc = self.featurers.initialization_output()
        k_pert = self.featurers.k_perturb+1
        ks = list(range(k_pert))
        for iss, neighs_info in self.retrievers:
            characs_iss, vals_iss =\
                self.featurers.compute_descriptors(iss, neighs_info, ks)
            desc = self.featurers.add2result(desc, characs_iss, vals_iss)
        desc = self.featurers.to_complete_measure(desc)
        return desc

    def _compute_descriptors(self, i):
        "Compute the descriptors assigned to element i."
#        print 'b'*10, i
        staticneighs, typeret, typefeats = self._get_methods(i)
#        print 'c', i
        k_pert = self.featurers.k_perturb+1
        ks = list(range(k_pert))
        neighs_info = self.retrievers.retrieve_neighs(i, typeret_i=typeret)
        neighs_info.set_ks(ks)
        ## TESTING ASSERTIONS
#        assert(staticneighs == neighs_info.staticneighs)
#        i_len = 1 if type(i) == int else len(i)
#        i_list = [i] if type(i) == int else i
#        print 'd', i
#        print i_len, ks, neighs_info.iss, neighs_info.ks
#        print neighs_info.idxs
#        assert(len(neighs_info.iss) == i_len)
#        assert(neighs_info.iss == i_list)
#        if not staticneighs:
#            assert(len(neighs_info.ks) == len(ks))
#            assert(neighs_info.ks == ks)
#        print 'a'*25, typefeats, typeret, i
        #####################
        characs, vals_i =\
            self.featurers.compute_descriptors(i, neighs_info, ks, typefeats)
        return characs, vals_i

#    def _compute_descriptors(self, i):
#        "Compute the descriptors assigned to element i."
#        staticneighs, typeret, typefeats = self._get_methods(i)
#        if staticneighs:
#            characs, vals_i = self._compute_descriptors_seq0(i, typeret,
#                                                             typefeats)
#        else:
#            characs, vals_i = self._compute_descriptors_seq1(i, typeret,
#                                                             typefeats)
#
#        return characs, vals_i
#
#    def _compute_descriptors_seq0(self, i, typeret, typefeats):
#        "Computation descriptors for non-aggregated data."
#        ## Model1
#        staticneighs, _, _ = self._get_methods(i)
#        k_pert = self.featurers.k_perturb+1
#        ks = list(range(k_pert))
#        neighs_info =\
#            self.retrievers.retrieve_neighs(i, typeret_i=typeret)  #, k=ks)
#        assert(staticneighs == neighs_info.staticneighs)
#        characs, vals_i =\
#            self.featurers.compute_descriptors(i, neighs_info, ks, typefeats)
#        return characs, vals_i
#
#    def _compute_descriptors_seq1(self, i, typeret, typefeats):
#        "Computation descriptors for aggregated data."
#        k_pert = self.featurers.k_perturb+1
#        characs, vals_i = [], []
#        for k in range(k_pert):
#            neighs_info =\
#                self.retrievers.retrieve_neighs(i, typeret_i=typeret, k=k)
#            assert(len(neighs_info.ks) == 1)
#            assert(neighs_info.ks[0] == k)
#            characs_k, vals_i_k =\
#                self.featurers.compute_descriptors(i, neighs_info,
#                                                   k, typefeats)
#            characs.append(characs_k)
#            vals_i.append(vals_i_k)
#        ## Joining descriptors from different perturbations
#        characs = self.featurers._join_descriptors(characs)
#        vals_i = np.concatenate(vals_i)
#        return characs, vals_i

    ################################ ITERATORS ################################
    ###########################################################################
    def compute_nets_i(self):
        """Computation of the associate spatial descriptors for each i."""
        for i in self.iter_indices():
            ## Compute descriptors for i
            desc_i, vals_i = self._compute_descriptors(i)
            yield desc_i, vals_i

    def compute_net_ik(self):
        """Function iterator used to get the result of each val_i and corr_i
        result for each combination of element i and permutation k.
        """
        for i in self.iter_indices():
            for k in range(self.retrievers.k_perturb+1):
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


###############################################################################
############################# Auxiliar functions ##############################
###############################################################################
############################# Create aggfeatures ##############################
###############################################################################
def create_aggfeatures(sp_descriptor, features):
    """Average distance of points of the different regions.
    Function to compute the spatial distances between regions.

    Parameters
    ----------
    sp_descriptor: tuple (aggregation_info format) or SpatialDescriptorModel
        the information to compute aggregation.

    Returns
    -------
    new_exlicit_features: pst.Features object
        the features object obtained by aggregating the features.

    """
    ## 1. Computing
    if type(sp_descriptor) == tuple:
        ## 0. Parsing inputs
        disc_info, retriever_in, _, agg_info = sp_descriptor
        assert(type(agg_info) == tuple)
        assert(len(agg_info) == 2)
        _, aggregating_feat = agg_info
        agg_f_ret, desc_in, pars_feat_in, pars_feats, desc_out =\
            _parse_aggregation_feat(aggregating_feat, features)
        ## Retrievers
        locs, regs, disc = _discretization_parsing_creation(disc_info)
        retrievers = agg_f_ret(retriever_in, locs, regs, disc)
        ## Featurers
        # Feature creation
        object_feats, core_features, pars_fea_o_in = features.export_features()
        pars_fea_o_in['descriptormodel'] = desc_in
        new_features = object_feats(core_features, **pars_fea_o_in)
        # Feature manager creation
        pars_feat_in['maps_vals_i'] = regs
        pars_feat_in['selectors'] = (0, 0), (0, 0), (0, 0)
        featurers = FeaturesManager(new_features, **pars_feat_in)
        ## 1. Preparing spdesc object
        spdesc = SpatialDescriptorModel(retrievers, featurers)
        ## 2. Compute
        aggfeatures = spdesc.compute()
    else:
        aggfeatures = sp_descriptor.compute()
        pars_feats = {}
        desc_out = sp_descriptor.featurers.features[0].descriptormodel

    ## 3. Creation of the object features aggregations
    new_exlicit_features =\
        _featuresobject_parsing_creation((aggfeatures, pars_feats, desc_out))
    return new_exlicit_features


def _parse_aggregation_feat(aggregating_in, features):
    assert(type(aggregating_in) == tuple)
    if len(aggregating_in) == 5:
        agg_f_ret, desc_in, pars_feat_in, pars_feats, desc_out = aggregating_in
    elif len(aggregating_in) == 4:
        agg_f_ret, desc_in, pars_feat_in, pars_feats = aggregating_in
        desc_out = features.descriptormodel

    elif len(aggregating_in) == 3 and type(aggregating_in[1]) == dict:
        agg_f_ret, pars_feat_in, pars_feats = aggregating_in
        desc_in = features.descriptormodel
        desc_out = features.descriptormodel
    elif len(aggregating_in) == 3 and type(aggregating_in[1]) != dict:
        agg_f_ret, desc_in, desc_out = aggregating_in
        pars_feat_in, pars_feats = {}, {}
    else:
        agg_f_ret = aggregating_in[0]
        pars_feat_in, pars_feats = {}, {}
        desc_in = features.descriptormodel
        desc_out = features.descriptormodel
    return agg_f_ret, desc_in, pars_feat_in, pars_feats, desc_out
