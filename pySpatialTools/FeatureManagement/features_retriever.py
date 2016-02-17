
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

"""

import numpy as np
## Check initialization of map vals i
from ..utils.util_classes import create_mapper_vals_i
from aux_descriptormodels import append_addresult_function
from aux_featuremanagement import create_aggfeatures, compute_featuresnames
from features_objects import ImplicitFeatures, ExplicitFeatures


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
        self.descriptormodel = None
        self.features = []
        ## IO information
        self._variables = {}
        self.featuresnames = []
        self.k_perturb = 0
        self._out = 'ndarray'  # dict
        ## IO managers
        self._maps_input = []
        self._maps_output = None
        self._maps_vals_i = None

    def __init__(self, features_objects, descriptormodel, maps_input=None,
                 maps_output=None, maps_vals_i=None, out=None):
        self._initialization()
        out = out if out in ['ndarray', 'dict'] else None
        self._out = self._out if out is None else out
        self._format_features(features_objects)
        self._format_maps(maps_input, maps_output, maps_vals_i)
        self._format_descriptormodel(descriptormodel)

    def __getitem__(self, i_feat):
        if i_feat < 0 or i_feat >= len(self.features):
            raise IndexError("Not correct index for features.")
        return self.features[i_feat]

    def __len__(self):
        return len(self.features)

    @property
    def shape(self):
        return self.features[0].shape

    @property
    def nfeats(self):
        return len(self.featuresnames)

    ################################ Formatters ###############################
    ###########################################################################
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
        ## 1. Check input
        if nfeat == 0:
            msg = "Any feature object is input in the featureRetriever."
            raise TypeError(msg)
        ## 2. Get main global information from feature objects
        k_perturb = [features_objects[i].k_perturb for i in range(nfeat)]
        vars_o = [set(features_objects[i].variables) for i in range(nfeat)]
        k_rei_bool = [k_perturb[i] == k_perturb[0] for i in range(nfeat)]
        ## 3. Check class parameters
        # Check k perturbations
        if not all(k_rei_bool):
            msg = "Not all the feature objects have the same perturbations."
            raise Exception(msg)
        # Check variables
        ## 4. Storing variables
        self._variables = vars_o[0]
        self.k_perturb = k_perturb[0]
        self.features = features_objects
        for i in range(nfeat):
            self.features[i]._out = self._out

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

    def _format_descriptormodel(self, descriptormodel):
        """Formatter of the descriptormodel."""
        ## Check descriptormodel
        if descriptormodel is None:
            raise TypeError("Incorrect descriptormodel type.")
        ## Outfeatures management
        out_features = []
        for i in range(len(self)):
            out_i = compute_featuresnames(descriptormodel, self.features[i])
            out_features.append(out_i)
        logi = all([out == out_features[0] for out in out_features])
        if logi:
            self.featuresnames = out_features[0]
            for i in range(len(self)):
                self[i].out_features = out_features[0]
        else:
            raise Exception("Not the same output features.")
        ## Set descriptormodel
        self.descriptormodel = descriptormodel
        for i in range(len(self)):
            self.features[i].set_descriptormodel(descriptormodel)
        ## Set output
        self._format_result_building(descriptormodel)

    def _format_map_vals_i(self, sp_typemodel):
        """Format mapper to indicate external val_i to aggregate result."""
        if sp_typemodel is not None:
            if type(sp_typemodel) == tuple:
                map_vals_i = create_mapper_vals_i(sp_typemodel, self)
            else:
                map_vals_i = create_mapper_vals_i(sp_typemodel, self)
            self._maps_vals_i = map_vals_i
        else:
            self._maps_vals_i = create_mapper_vals_i(self._maps_vals_i, self)

    def _format_maps(self, maps_input, maps_output, maps_vals_i):
        "Formatter of maps."
        ## 1. Format input maps
        if maps_input is None:
            self._maps_input = [lambda i, k=0: (i, k)]
        else:
            if type(maps_input).__name__ == 'function':
                self._maps_input = [lambda i, k=0: maps_input(i, k)]
            else:
                self._maps_input = [maps_input]
        ## 2. Format output maps (TODO)
        if maps_output is None:
            self._maps_output = lambda self, feats: feats
        else:
            if type(maps_output).__name__ == 'function':
                self._maps_output = lambda feats: maps_output(self, feats)
            else:
                self._maps_output = maps_output
        self._format_map_vals_i(maps_vals_i)

    def _format_result_building(self, descriptormodel):
        """Format how to build and aggregate results."""
        "TODO: Dict-array."
        "TODO: null_value"
        ## Size of the possible results.
        n_vals_i = self._maps_vals_i.n_out
        n_feats = self.nfeats
        ## Initialization features
        if self._out == 'ndarray':
            self.initialization_desc = lambda: np.zeros((1, n_feats))
        else:
            ## TODO:
            pass
        ## Global construction of result
        if n_vals_i is not None and self._out == 'ndarray':
            shape = (n_vals_i, n_feats, self.k_perturb + 1)
            ## Init global result
            self.initialization_output = lambda: np.zeros(shape)
            ## Adding result
            self.add2result = descriptormodel._defult_add2result
        else:
            ## Init global result
            self.initialization_output = lambda: []
            ## Adding result
            self.add2result = append_addresult_function
        self.to_complete_measure =\
            lambda X: descriptormodel.to_complete_measure(X)

    ################################# Setters #################################
    ###########################################################################
    def set_map_vals_i(self, _maps_vals_i):
        "Set how it maps each element of the "
        #self._maps_vals_i = _maps_vals_i
        if type(_maps_vals_i) in [int, slice]:
            _maps_vals_i = (self._maps_vals_i, _maps_vals_i)
            self._format_map_vals_i(_maps_vals_i)
        self._format_map_vals_i(_maps_vals_i)

    ################################# Getters #################################
    ###########################################################################
    def compute_descriptors(self, i, neighs_info, k=None, feat_selectors=None):
        """General compute descriptors for descriptormodel class.
        """
        ## 0. Prepare list of k
        ks = list(range(self.k_perturb+1)) if k is None else k
        ks = [ks] if type(ks) == int else ks
        ## 1. Prepare selectors
        t_feat_in, t_feat_out = self._get_typefeats(feat_selectors)
        ## 2. Get pfeats (pfeats 2dim array (krein, jvars))
        desc_i = self._get_input_features(i, ks, t_feat_in)
        desc_neigh = self._get_output_features(neighs_info, ks, t_feat_out)
        ## 3. Map vals_i
        vals_i = self._get_vals_i(i, ks)

        ## 4. Complete descriptors
        descriptors =\
            self.descriptormodel.complete_desc_i(i, neighs_info, desc_i,
                                                 desc_neigh, vals_i)
        return descriptors, vals_i

    def _get_input_features(self, i, k, typefeats=(0, 0)):
        """Get 'input' features. Get the features of the elements of which we
        want to study their neighbourhood.
        """
        ## Retrieve features
        if type(i) == tuple:
            i_input, k_input = self._maps_input[typefeats[0]](i[0], k)
            i_input = i_input, i[1]
        else:
            i_input, k_input = self._maps_input[typefeats[0]](i, k)
        feats_i = self.features[typefeats[1]][i_input, k_input]
        ## Outformat
        feats_i = self._maps_output(self, feats_i)
        return feats_i

    def _get_output_features(self, idxs, k, typefeats=(0, 0)):
        """Get 'output' features. Get the features of the elements in the
        neighbourhood of the elements we want to study."""
        ## Retrieve features
        if type(idxs) == tuple:
            idxs_input, k_input = self._maps_input[typefeats[0]](idxs[0], k)
            idxs_input = idxs_input, idxs[1]
        else:
            idxs_input, k_input = self._maps_input[typefeats[0]](idxs, k)
        if np.any(idxs_input):
            feats_idxs = self.features[typefeats[1]][idxs_input, k_input]
        else:
            null_value = self.features[typefeats[1]]._nullvalue
            feats_idxs = np.ones((len(k_input), self.shape[1])) * null_value
        ## Outformat
        feats_idxs = self._maps_output(self, feats_idxs)
        return feats_idxs

    def _get_vals_i(self, i, ks):
        """Get indice to store the final result."""
        ## 0. Prepare variable needed
        vals_i = []
        ## 1. Loop over possible ks and compute vals_i
        for k in ks:
            vals_i.append(self._maps_vals_i.apply(self, i, k))
        vals_i = np.array(vals_i).ravel()
        return vals_i

    def _get_typefeats(self, typefeats):
        """Format properly typefeats selector information."""
        if typefeats is None:
            typefeats = (0, 0, 0, 0)
        if type(typefeats) != tuple:
            typefeats = (0, 0, 0, 0)
        elif len(typefeats) != 4:
            typefeats = (0, 0, 0, 0)
        return typefeats[:2], typefeats[2:]

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
