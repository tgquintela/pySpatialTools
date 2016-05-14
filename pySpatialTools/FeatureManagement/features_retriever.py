
"""
Feature Retriever
-----------------
Information about features which groups all together in the homogenious ouput
of features, but they can have heterogenous input as it is retrieved by the
element retrievers.
This module contains the tools to store and retrieve features from a pool of
features, within the descriptormodel we use to compute descriptors from
element features.


Check: same type variables output
Check: same k dimension


TODO
----
- Indentify different type of feaures we can have: element, aggregate...
- Support for other type of features?
- Support for dictionary
- Or all the same features and choose one or all different and join together.

"""

import numpy as np
## Check initialization of map vals i
from ..utils.util_classes import create_mapper_vals_i,\
    Feat_RetrieverSelector, ensuring_neighs_info

from aux_descriptormodels import append_addresult_function,\
    replacelist_addresult_function, sparse_dict_completer,\
    sparse_dict_completer_unknown, sum_addresult_function
from aux_featuremanagement import create_aggfeatures
from features_objects import ImplicitFeatures, ExplicitFeatures
from descriptormodel import DummyDescriptor


class FeaturesManager:
    """Method for retrieving features.

    See also
    --------
    pySpatialTools.RetrieverManager

    """
    __name__ = "pySpatialTools.FeaturesManager"
    typefeat = "manager"

    def _initialization(self):
        """Initialization of mutable class parameters."""
        ## Main objects to manage
        self.descriptormodels = []
        self.features = []
        self.mode = None
        self.selector = (0, 0), (0, 0), (0, 0)
        ## IO information
        self._variables = {}       ## TO CHANGE
        self.featuresnames = []    ## TO CHANGE
        self.out_features = []
        self.k_perturb = 0
        self._out = 'ndarray'  # dict
        ## IO managers
        self._maps_input = []
        self._maps_output = None
        self._maps_vals_i = None

    def __init__(self, features_objects, mode=None, maps_input=None,
                 maps_output=None, maps_vals_i=None, descriptormodels=None,
                 selectors=[None]*3):
        self._initialization()
#        out = out if out in ['ndarray', 'dict'] else None
#        self._out = self._out if out is None else out
        self._format_maps(maps_input, maps_output, maps_vals_i)
        self._format_features(features_objects)
        self._format_mode(mode)
        self._format_outfeatures()
        self._format_result_building()
        self._format_descriptormodel(descriptormodels)
        self.set_selector(*selectors)

    def __getitem__(self, i_feat):
        if i_feat < 0 or i_feat >= len(self.features):
            raise IndexError("Not correct index for features.")
        return self.features[i_feat]

    def __len__(self):
        return len(self.features)

    @property
    def shape(self):
        """As a mapper its shapes represents the size of the input and the
        output stored in maps_vals_i."""
        return self._maps_vals_i.n_in, self._maps_vals_i.n_out

    @property
    def nfeats(self):
        return len(self.featuresnames)

    ################################ Formatters ###############################
    ###########################################################################
    ############################## Format features ############################
    def _format_features(self, features_objects):
        """Formatter of features."""
        ## 0. Format to list mode
        # Format to list mode
        if type(features_objects) != list:
            features_objects = [features_objects]
        # Format to feature objects
        nfeat = len(features_objects)
        for i in range(nfeat):
            features_objects[i] = self._auxformat_features(features_objects[i])
        self.features = features_objects
        ## 1. Check input
        if nfeat == 0:
            msg = "Any feature object is input in the featureRetriever."
            raise TypeError(msg)
        ## 2. Format kperturb
        kp = self[0].k_perturb
        k_rei_bool = [self[i].k_perturb == kp for i in range(len(self))]
        # Check k perturbations
        if not all(k_rei_bool):
            msg = "Not all the feature objects have the same perturbations."
            raise Exception(msg)
        self.k_perturb = kp

    def _auxformat_features(self, features):
        """Format individual features information."""
        if type(features) == np.ndarray:
            sh = features.shape
            if len(sh) == 1:
                features = features.reshape((sh[0], 1))
                features = ImplicitFeatures(features)
            if len(sh) == 2:
                features = ImplicitFeatures(features)
            elif len(sh) == 3:
                features = ExplicitFeatures(features)
        else:
            if not features.__name__ == "pySpatialTools.FeaturesObject":
                raise TypeError("Incorrect features format.")
        return features

    def _format_descriptormodel(self, descriptormodels=None):
        """Formatter of the descriptormodels."""
        if descriptormodels is None:
            self.descriptormodels = [DummyDescriptor()]
        else:
            if type(descriptormodels) != list:
                descriptormodels = [descriptormodels]
            self.descriptormodels += descriptormodels

    def _format_result_building(self):
        ## Size of the possible results.
        n_vals_i = self._maps_vals_i.n_out
        n_feats = len(self.out_features)
        if self._out == 'ndarray':
            assert(self.mode in ['sequential', 'parallel'])

            ## Format initialization descriptors
            def initialization_desc():
                descriptors = []
                for i in range(len(self)):
                    aux_i = np.ones((1, len(self[i].out_features)))
                    descriptors.append(aux_i * self[i]._nullvalue)
                descriptors = np.concatenate(descriptors, axis=1)
                return descriptors
            # Set initialization descriptors
            self.initialization_desc = initialization_desc
            ## Pure Array
            if n_vals_i is not None:
                shape = (n_vals_i, n_feats, self.k_perturb+1)
                # Set initialization output descriptors
                self.initialization_output = lambda: np.zeros(shape)
                self._join_descriptors = lambda x: np.concatenate(x)
                ## Option to externally set add2result and to_complete
                self.add2result = sum_addresult_function
                self.to_complete_measure = lambda X: X
            ## List array measure
            else:
                shape = (n_vals_i, n_feats, self.k_perturb+1)
                # Set initialization output descriptors
                self.initialization_output = lambda: []  ## TODO: All dimensions
                self._join_descriptors = lambda x: np.concatenate(x)
                ## Option to externally set add2result and to_complete
                self.add2result = append_addresult_function
                self.to_complete_measure = lambda X: np.concatenate(X)
        else:
            self.initialization_desc = lambda: [{}]
            if n_vals_i is not None:
                ## Init global result
                self.initialization_output =\
                    lambda: [[[] for i in range(n_vals_i)]
                             for k in range(self.k_perturb+1)]
                self._join_descriptors = lambda x: x
                ## Adding result (TODO: External set option)
                self.add2result = append_addresult_function
                self.to_complete_measure = sparse_dict_completer
            else:
                ## Init global result
                self.initialization_output =\
                    lambda: [[[], []] for k in range(self.k_perturb+1)]
                self._join_descriptors = lambda x: x
                ## Adding result (TODO: External set option)
                self.add2result = replacelist_addresult_function
                self.to_complete_measure = sparse_dict_completer_unknown

    ############################## Format IO maps #############################
    def _format_map_vals_i(self, sp_typemodel):
        """Format mapper to indicate external val_i to aggregate result."""
        if sp_typemodel is not None:
            map_vals_i = create_mapper_vals_i(sp_typemodel, self)
#            if type(sp_typemodel) == tuple:
#                map_vals_i = create_mapper_vals_i(sp_typemodel, self)
#            else:
#                map_vals_i = create_mapper_vals_i(sp_typemodel, self)
            self._maps_vals_i = map_vals_i
        else:
            self._maps_vals_i = create_mapper_vals_i(self._maps_vals_i, self)

    def _format_maps(self, maps_input, maps_output, maps_vals_i):
        "Formatter of maps."
        ## 1. Format input maps
        if maps_input is None:
            self._maps_input = [lambda i_info, k=0: i_info]
        else:
            if type(maps_input).__name__ == 'function':
                self._maps_input = [lambda i_info, k=0: maps_input(i_info, k)]
#            else:
#                self._maps_input = [maps_input]
        ## 2. Format output maps (TODO)
        if maps_output is None:
            self._maps_output = lambda self, feats: feats
        else:
            if type(maps_output).__name__ == 'function':
                self._maps_output = lambda s, feats: maps_output(s, feats)
#            else:
#                self._maps_output = maps_output
        self._format_map_vals_i(maps_vals_i)

    ############################ Format mode utils ############################
    def _format_mode(self, mode=None):
        """Format which mode is going to be this manager (sequential or
        parallel). In sequential the features are options between which we
        have to choose in order to get the outfeatures. In parallel we have to
        use everyone and join the results into a final result.
        """
        ## Format basic modes
        if mode is None:
            ## If open out_features
            if not len(self[0].out_features):
                self._out = 'dict'
                n_f = len(self)
                logi = [not len(self[i].out_features) for i in range(n_f)]
                assert(all(logi))
                self.out_features = []
                self.mode = None
            else:
                self._out = 'ndarray'
                outs = self[0].out_features
                logi = [self[i].out_features == outs for i in range(len(self))]
                if all(logi):
                    n_f = len(self)
                    nulls = self[0]._nullvalue
                    assert([self[i]._nullvalue == nulls for i in range(n_f)])
                    self.mode = 'sequential'
                else:
                    self.mode = 'parallel'
        else:
            assert(mode in ['sequential', 'parallel'])
            self.mode = mode

    def _format_outfeatures(self):
        """Format the output features of this manager."""
        ## If open out_features
        logi = [type(self[i].out_features) == list for i in range(len(self))]
        assert(all(logi))
        if len(self[0].out_features) == 0:
            logi = [len(self[i].out_features) == 0 for i in range(len(self))]
            assert(all(logi))
            self.out_features = []
        ## If close out_features
        else:
            ## If mode parallel
            if self.mode == 'parallel':
                outfeatures = []
                for i in range(len(self)):
                    outfeatures.append(self[i].out_features)
                self.out_features = outfeatures
            ## If mode sequential
            elif self.mode == 'sequential':
                outs = self[0].out_features
                logi = [self[i].out_features == outs for i in range(len(self))]
                assert(all(logi))
                self.out_features = outs

    ############################# Format selectors ############################
    def _format_selector(self, selector1, selector2=None, selector3=None):
        """Programable get_type_feats."""
        if selector1 is None:
            self.get_type_feats = self._general_get_type_feat
            self._get_input_features = self._get_input_features_general
            self._get_output_features = self._get_output_features_general
            self._complete_desc_i = self._complete_desc_i_general
        else:
            typ = type(selector1)
            if selector2 is not None:
                assert((type(selector2) == typ) and (type(selector2) == typ))
            if typ == tuple:
                self.selector = (selector1, selector2, selector3)
                self.get_type_feats = self._static_get_type_feat
                self._get_input_features = self._get_input_features_constant
                self._get_output_features = self._get_output_features_constant
                self._complete_desc_i = self._complete_desc_i_constant
            else:
                self.selector =\
                    Feat_RetrieverSelector(selector1, selector2, selector3)
                self.get_type_feats = self._selector_get_type_feat
                self._get_input_features = self._get_input_features_variable
                self._get_output_features = self._get_output_features_variable
                self._complete_desc_i = self._complete_desc_i_variable
                self.selector.assert_correctness(self)

    ################################# Setters #################################
    ###########################################################################
    def set_map_vals_i(self, _maps_vals_i):
        "Set how it maps each element of the "
        #self._maps_vals_i = _maps_vals_i
        if type(_maps_vals_i) in [int, slice]:
            _maps_vals_i = (self._maps_vals_i, _maps_vals_i)
            self._format_map_vals_i(_maps_vals_i)
        self._format_map_vals_i(_maps_vals_i)

    def set_descriptormodels(self, descriptormodels):
        """Set descriptormodels."""
        self._format_descriptormodel(descriptormodels)

    def set_selector(self, selector1, selector2=None, selector3=None):
        """Set selectors."""
        self._format_selector(selector1, selector2, selector3)

    ################################# Getters #################################
    ###########################################################################
    def compute_descriptors(self, i, neighs_info, k=None, feat_selectors=None):
        """General compute descriptors for descriptormodel class.
        """
        ## 0. Prepare list of k
        ks = list(range(self.k_perturb+1)) if k is None else k
        ks = [ks] if type(ks) == int else ks
        i_input = [i] if type(i) == int else i
        neighs_info = ensuring_neighs_info(neighs_info, k)
#        sh = neighs_info.shape
#        print sh, len(i_input), len(ks), ks, i_input
#        assert(len(i_input) == sh[1])
#        assert(len(ks) == sh[0])
        ## 1. Prepare selectors
        t_feat_in, t_feat_out, t_feat_des =\
            self.get_type_feats(i_input, feat_selectors)
        ## 2. Get pfeats (pfeats 2dim array (krein, jvars))
        desc_i = self._get_input_features(i_input, ks, t_feat_in)
        desc_neigh = self._get_output_features(neighs_info, ks, t_feat_out)
#        print i, ks, i_input, neighs_info, neighs_info.ks, neighs_info.idxs
        ## 3. Map vals_i
        vals_i = self._get_vals_i(i, ks)
#        print '+'*10, vals_i, desc_neigh, desc_i

        ## 4. Complete descriptors (TODO)
        descriptors = self._complete_desc_i(i, neighs_info, desc_i, desc_neigh,
                                            vals_i, t_feat_des)
#        descriptors =\
#            self.descriptormodel.complete_desc_i(i, neighs_info, desc_i,
#                                                 desc_neigh, vals_i)
        ## 5. Join descriptors
        descriptors = self._join_descriptors(descriptors)

        return descriptors, vals_i

    ######################## Interaction with features ########################
    ############################# Input features ##############################
    def _get_input_features_constant(self, i, k, typefeats=(0, 0)):
        """Get 'input' features. Get the features of the elements of which we
        want to study their neighbourhood. We consider constant typefeats.
        """
        ## Input mapping
        i_input = self._maps_input[typefeats[0]](i)
        ## Retrieve features
        feats_i = self.features[typefeats[1]][i_input, k]
        ## Outformat
        feats_i = self._maps_output(self, feats_i)
        return feats_i

    def _get_input_features_variable(self, i, k, typefeats=(0, 0)):
        """Get 'input' features. Get the features of the elements of which we
        want to study their neighbourhood. We consider constant typefeats.
        """
        ## Preparing input
        i = [[i]] if type(i) == int else i
        i_l = len(i)
        k_l = 1 if type(k) == int else len(k)
        typefeats = [typefeats]*i_l if type(typefeats) == tuple else typefeats
        feats_i = [[] for kl in range(k_l)]
        for j in range(i_l):
            ## Input mapping
            i_j = [i[j]] if type(i[j]) == int else i[j]
            i_input = self._maps_input[typefeats[j][0]](i_j)
            ## Retrieve features
            feats_ij = self.features[typefeats[j][1]][i_input, k]
            ## Outformat
            feats_ij = self._maps_output(self, feats_ij)
            for k_j in range(len(feats_ij)):
                feats_i[k_j].append(feats_ij[k_j][0])
        assert(len(feats_i) == k_l)
        assert(len(feats_i[0]) == i_l)
        return feats_i

    def _get_input_features_general(self, i, k, typefeats=(0, 0)):
        """Get 'input' features. Get the features of the elements of which we
        want to study their neighbourhood. We dont consider anything.
        """
        if type(typefeats) == list:
            feats_i = self._get_input_features_variable(i, k, typefeats)
        else:
            feats_i = self._get_input_features_constant(i, k, typefeats)
        return feats_i

    ############################# Output features #############################
    def _get_output_features_constant(self, neighs_info, k, typefeats=(0, 0)):
        """Get 'output' features. Get the features of the elements in the
        neighbourhood of the elements we want to study."""
        ## Neighs info as an object
        neighs_info = ensuring_neighs_info(neighs_info, k)
        ## Input mapping
        neighs_info = self._maps_input[typefeats[0]](neighs_info)
        ## Features retrieve
        feats_neighs = self.features[typefeats[1]][neighs_info, k]
        ## Outformat
        feats_neighs = self._maps_output(self, feats_neighs)
        return feats_neighs

    def _get_output_features_variable(self, neighs_info, k, typefeats=(0, 0)):
        """Get 'output' features. Get the features of the elements in the
        neighbourhood of the elements we want to study."""
        ## Neighs info as an object
        neighs_info = ensuring_neighs_info(neighs_info, k)
        ## Loop for all typefeats
        i_l = len(neighs_info.iss)
#        print i_l, neighs_info.iss, neighs_info.idxs
        k_l = 1 if type(k) == int else len(k)
        typefeats = [typefeats]*i_l if type(typefeats) == tuple else typefeats
        feats_neighs = [[] for kl in range(k_l)]
        for j in range(i_l):
            ## Input mapping
            neighs_info_j = neighs_info.get_copy_iss_by_ind(j)
            neighs_info_j = self._maps_input[typefeats[j][0]](neighs_info_j)
            ## Features retrieve
            feats_neighs_j = self.features[typefeats[j][1]][neighs_info_j, k]
            ## Outformat
            feats_neighs_j = self._maps_output(self, feats_neighs_j)
            ## Store
            for k_j in range(len(feats_neighs_j)):
                feats_neighs[k_j].append(feats_neighs_j[k_j][0])
        assert(len(feats_neighs) == k_l)
        assert(len(feats_neighs[0]) == i_l)
        return feats_neighs

    def _get_output_features_general(self, neighs_info, k, typefeats=(0, 0)):
        """Get 'output' features. Get the features of the elements in the
        neighbourhood of the elements we want to study."""
        if type(typefeats) == list:
            feats_neighs =\
                self._get_output_features_variable(neighs_info, k, typefeats)
        else:
            feats_neighs =\
                self._get_output_features_constant(neighs_info, k, typefeats)
        return feats_neighs

    ########################### Descriptor features ###########################
    def _complete_desc_i_constant(self, i, neighs_info, desc_i, desc_neigh,
                                  vals_i, t_feat_desc):
        """Complete descriptors by interaction of point features and
        neighbourhood features."""
        if t_feat_desc[0]:
            descriptors = self.features[t_feat_desc[1]].\
                complete_desc_i(i, neighs_info, desc_i, desc_neigh, vals_i)
        else:
            descriptors = self.descriptormodels[t_feat_desc[1]].\
                complete_desc_i(i, neighs_info, desc_i, desc_neigh, vals_i)
        return descriptors

    def _complete_desc_i_variable(self, i, neighs_info, desc_i, desc_neigh,
                                  vals_i, t_feat_desc):
        """Complete descriptors by interaction of point features and
        neighbourhood features. Assumption of variable t_feat_desc."""
        ## Preparing input
        i_l = 1 if type(i) == int else len(i)
        i = [i]*i_l if type(i) == int else i
        k_l = len(desc_i)
        if type(t_feat_desc) != list:
            t_feat_desc = [t_feat_desc]*i_l
        ## Sequential computation
        descriptors = [[] for kl in range(k_l)]
#        print 'joe', i_l, len(desc_i), len(desc_neigh), len(vals_i)
        for j in range(i_l):
            neighs_info_j = neighs_info.get_copy_iss_by_ind(j)
            vals_ij = [vals_i[k][j] for k in range(len(vals_i))]
            desc_ij = [desc_i[k][j] for k in range(len(desc_i))]
            desc_neighj = [desc_neigh[k][j] for k in range(len(desc_neigh))]

            if t_feat_desc[j][0]:
                descriptors_j = self.features[t_feat_desc[j][1]].\
                    complete_desc_i(i[j], neighs_info_j,
                                    desc_ij, desc_neighj, vals_ij)
            else:
                descriptors_j = self.descriptormodels[t_feat_desc[j][1]].\
                    complete_desc_i(i[j], neighs_info_j,
                                    desc_ij, desc_neighj, vals_ij)
            for k_j in range(k_l):
                descriptors[k_j].append(descriptors_j[k_j])
        return descriptors

    def _complete_desc_i_general(self, i, neighs_info, desc_i, desc_neigh,
                                 vals_i, t_feat_desc):
        """Complete descriptors by interaction of point features and
        neighbourhood features. No assumptions about iss and t_feat_desc."""
        if type(t_feat_desc) == list:
            descriptors =\
                self._complete_desc_i_variable(i, neighs_info, desc_i,
                                               desc_neigh, vals_i, t_feat_desc)
        else:
            descriptors =\
                self._complete_desc_i_constant(i, neighs_info, desc_i,
                                               desc_neigh, vals_i, t_feat_desc)
        return descriptors

    ############################### Map_vals_i  ###############################
    def _get_vals_i(self, i, ks):
        """Get indice to store the final result."""
        #### TODO: extend
        ## 0. Prepare variable needed
        vals_i = []
        ## 1. Loop over possible ks and compute vals_i
        for k in ks:
            vals_i.append(self._maps_vals_i.apply(self, i, k))
        ## WARNING: TOTEST
        vals_i = np.array(vals_i)
#        print vals_i
        assert(len(vals_i.shape) == 2)
        assert(len(vals_i) == len(ks))
        assert(len(np.array([i]).ravel()) == vals_i.shape[1])
        return vals_i

    ################################# Type_feat ###############################
    ###########################################################################
    ## Formatting the selection of path from i information for features
    ## retrieving.
    ##
    ## See also:
    ## ---------
    ## pst.RetrieverManager
    #########################
    def _general_get_type_feat(self, i, typefeats_i=None):
        """Format properly general typefeats selector information."""
        if typefeats_i is None or type(typefeats_i) != tuple:
            typefeats_i, typefeats_nei, typefeats_desc = self.selector
        else:
            typefeats_i, typefeats_nei, typefeats_desc = typefeats_i
        return typefeats_i, typefeats_nei, typefeats_desc

    def _static_get_type_feat(self, i, typefeats_i=None):
        """Format properly typefeats selector information."""
        typefeats_i, typefeats_nei, typefeats_desc = self.selector
        return typefeats_i, typefeats_nei, typefeats_desc

    def _selector_get_type_feat(self, i, typefeats_i=None):
        """Get information only from selector."""
        typefeats_i, typefeats_nei, typefeats_desc = self.selector[i]
        return typefeats_i, typefeats_nei, typefeats_desc

    ######################## Perturbation management ##########################
    ###########################################################################
    def add_perturbations(self, perturbations):
        """Adding perturbations to features.
        """
        for i_ret in range(len(self.features)):
            self.features[i_ret].add_perturbations(perturbations)

    ######################### Aggregation management ##########################
    ###########################################################################
    def add_aggregations(self, discretization, regmetric, retriever=None,
                         pars_features={}, kfeat=0):
        """Add aggregations to featuremanager. Only it is useful this function
        if there is only one retriever previously and we are aggregating the
        first one.
        """
        ## 0. Get kfeats
        kfeats = [kfeat] if type(kfeat) == int else kfeat
        if kfeats is None:
            kfeats = []
            for i in range(len(self)):
                if self.features[i].typefeat == 'implicit':
                    kfeats.append(i)
        ## 1. Compute and store aggregations
        for i in kfeats:
            aggfeatures = create_aggfeatures(discretization, regmetric,
                                             self.features[i],
                                             self.descriptormodel)
            self.features.append(aggfeatures)

    ####################### Auxiliar temporal functions #######################
    ###########################################################################
