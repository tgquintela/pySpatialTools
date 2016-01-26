
"""
Feature Retriever
-----------------
Information about features which groups all together in the homogenious ouput
of features, but they can have heterogenous input as it is retrieved by the
element retrievers.
This module contains the tools to store and retrieve features from a pool of
features.


Check: same type variables output
Check: same k dimension


TODO
----
- Indentify different type of feaures we can have: point, aggregate...

"""

import numpy as np


class FeaturesRetriever:
    "Method for retrieving features."

    features = []
    featuresnames = []
    _k_reindices = 1
    _variables = {}
    _maps_input = None
    _maps_output = None
    _out = 'ndarray'  # dict
    __name__ = "pst.FeaturesRetriever"

    def __init__(self, features_objects, maps_input=None, maps_output=None,
                 out=None):
        out = out if out in ['ndarray', 'dict'] else None
        self._out = self._out if out is None else out
        self._format_features(features_objects)
        self._format_maps(maps_input, maps_output)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i):
        return self.features[i]

    def apply_reindice(self, i, k):
        #### TODO: Temporal!!!! Comunicate with sp_descmodel.
        return i

    def set_descriptormodel(self, descriptormodel):
        ## We are assuming feature 0 is the representative one.
        self.featuresnames =\
            descriptormodel._compute_featuresnames(self.features[0])
        ## Set out_features
        for i in range(len(self)):
            out_feat = descriptormodel._compute_featuresnames(self.features[i])
            self.features[i].out_features = out_feat
#        self.compute_prefeatures = descriptormodel.compute_prefeatures
        init_ = np.ones(self.nfeats) * descriptormodel._nullvalue

        def init_features(_out):
            if _out == 'dict':
                return dict(zip(self.featuresnames, init_))
            elif _out == 'ndarray':
                return init_
        self.initialization_features = init_features

    @property
    def nfeats(self):
        return len(self.featuresnames)

    def _format_features(self, features_objects):
        "Formatter of features."
        ## Check variables
        if type(features_objects) != list:
            features_objects = [features_objects]
        nfeat = len(features_objects)
        k_reindices = [features_objects[i]._k_reindices for i in range(nfeat)]
        vars_o = [set(features_objects[i].variables) for i in range(nfeat)]
#        vars_o_bool = [vars_o[i] == vars_o[0] for i in range(nfeat)]
        k_rei_bool = [k_reindices[i] == k_reindices[0] for i in range(nfeat)]

#        ## Checkers (check later?)
#        if not all(vars_o_bool):
#            msg = "Not all the feature objects have the same variables."
#            raise Exception(msg)

        if not all(k_rei_bool):
            msg = "Not all the feature objects have the same reindices dim."
            raise Exception(msg)
        ## Storing variables
        self._variables = vars_o[0]
        self._k_reindices = k_reindices[0]
        self.features = features_objects
        for i in range(nfeat):
            self.features[i]._out = self._out

    def _format_maps(self, maps_input, maps_output):
        "Formatter of maps."
        if maps_input is None:
            self._maps_input = lambda i, k, typeret: (typeret, i, k)
        else:
            if type(maps_output).__name__ == 'function':
                self._maps_input =\
                    lambda i, k=0, typeret=0: maps_output(self, i, k, typeret)
            else:
                self._maps_input = maps_input
        if maps_output is None:
            self._maps_output = lambda i, k, typeret: (typeret, i, k)
        else:
            if type(maps_output).__name__ == 'function':
                self._maps_output =\
                    lambda i, k=0, typeret=0: maps_output(self, i, k, typeret)
            else:
                self._maps_output = maps_output

    def _get_input_features(self, i, k, typefeats):
        "Get input features."
        ## Retrieve features
        feat_o, i_input, k_input = self._maps_input(i, k, typefeats)
        feats_i = self.features[feat_o][i_input, k_input]
        return feats_i

    def _get_output_features(self, idxs, k, typefeats):
        "Get output features."
        ## Retrieve features
        feat_o, idxs_input, k_input = self._maps_output(idxs, k, typefeats)
        feats_idxs = self.features[feat_o][idxs_input, k_input]
        return feats_idxs

    def _get_prefeatures(self, i, neighs_info, k, typefeats):
        """General interaction with features object to get point features from
        it.
        """
        if k is None:
            desc_i, desc_neigh = [], []
            for k in range(self._k_reindices):
                desc_i = self._get_input_features(i, typefeats[0])
                desc_neigh = self._get_output_features(neighs_info[0], k,
                                                       typefeats[1])
            desc_i = np.vstack(desc_i)
            desc_neigh = np.vstack(desc_neigh)
        else:
            desc_i = self._get_input_features(i, k, typefeats[0])
            desc_neigh = self._get_output_features(neighs_info[0], k,
                                                   typefeats[1])
        return desc_i, desc_neigh


class DummyReindiceMapper:
    "Dummy mapper."
    reindices = None

    def __init__(self, reindices=None):
        self.reindices = reindices

    def __getitem__(self, key):
        i, k = key
        if self.reindices is None:
            return i
        else:
            return self.reindices[i, k]


class Features:
    "Features object."

    _out = 'ndarray'
    __name__ = 'pst.FeaturesObject'

    def __len__(self):
        return len(self.features)

    @property
    def shape(self):
        return (len(self.features), len(self.variables), self._k_reindices)

    def _format_out(self, feats):
        if type(feats).__name__ == self._out:
            return feats
        try:
            if type(feats) == dict:
                # so out==array
                feats_o = np.ones(len(self.out_features))*self._nullvalue
                for e in feats:
                    feats_o[list(self.out_features).index(e)] = feats[e]
                if len(feats_o.shape) == 1:
                    feats_o = feats_o.reshape((1, feats_o.shape[0]))
            elif type(feats) == np.ndarray:
                feats_o = dict(zip(self.out_features, feats.ravel()))
        except:
            raise Exception("Incorrect _out format.")
        return feats_o


class PointFeatures(Features):
    """Point features.

    TODO
    ----
    Support for other type of feature collections.

    """
    ## Main attributes
    features = None
    variables = None
    out_features = None
    ## Other attributes
    _nullvalue = 0
    _reindices = None
    _k_reindices = 0
    ## Function to homogenize output respect aggfeatures
    _characterizer = lambda s, x: x
    # Type
    _type = 'point'

    def __init__(self, features, reindices=None, names=[], out_features=[],
                 characterizer=None):
        self._format_reindices(reindices)
        self._format_features(features, out_features)
        self._format_characterizer(characterizer)
        self._format_variables(names)

    def __getitem__(self, key):
        # Format inputs
        print key
        try:
            i, k = key
        except:
            msg = "pst.Features __getitem__() takes exactly 2 indices."
            raise TypeError(msg)
        if type(i) == tuple:
            i, d = i
        else:
            d = None
        if type(i) == int:
            if i < 0 or i >= len(self.features):
                raise IndexError("Index out of bounds.")
            i = [i]
        if type(i) in [list, slice]:
            idxs = i
            idxs = [self._reindices[idxs[j], k] for j in range(len(idxs))]
        else:
            raise TypeError("Incorrect index type for pst.Features.")
        # Retrive features
        feats = self.features[idxs]
        feats = self._characterizer(feats, d)
        feats = self._format_out(feats)
        return feats

    def _format_reindices(self, reindices):
        "Format reindices."
        if reindices is None:
            self._reindices = DummyReindiceMapper()
            self._k_reindices = 1
        else:
            if type(reindices) == np.ndarray:
                if len(reindices.shape) == 1:
                    reindices = reindices.reshape((reindices.shape[0], 1))
                self._reindices = reindices
                self._k_reindices = reindices.shape[1]
            else:
                try:
                    reindices[0, 0]
                    self._reindices = reindices
                except:
                    raise Exception("Incorrect reindices.")

    def _format_features(self, features, out_features):
        "Format features."
        sh = features.shape
        features = features if len(sh) == 2 else features.reshape((sh[0], 1))
        self.features = features
        self.out_features = out_features

    def _format_characterizer(self, characterizer):
        """Format characterizer function. It is needed to homogenize outputs in
        order to have the same output type as the aggfeatures.
        """
        if characterizer is not None:
            self._characterizer = characterizer
        try:
            self[(0, 0), 0]
        except:
            raise TypeError("Incorrect characterizer.")

    def _format_variables(self, names):
        feats = self[(0, 0), 0]
        if names:
            self.variables = names
            if len(names) != feats.shape[1]:
                msg = """Not matching lengths of variable names and output
                    feats."""
                raise IndexError(msg)
        else:
            if type(feats) == dict:
                self.variables = feats.keys()
            else:
                n_feats = feats.shape[1]
                self.variables = list(range(n_feats))


class AggFeatures(Features):
    "Aggregate features class."

    ## Main attributes
    features = None
    variables = None
    out_features = None
    ## Other attributes
    _nullvalue = 0
    _reindices = None
    _k_reindices = 0
    ## Type
    _type = 'aggregated'

    def __init__(self, aggfeatures, names=[], nullvalue=None):
        self._format_aggfeatures(aggfeatures, names)
        self._nullvalue = self._nullvalue if nullvalue is None else nullvalue

    def __getitem__(self, key):
        # Format inputs
        try:
            i, k = key
        except:
            msg = "pst.Features __getitem__() takes exactly 2 indices."
            raise TypeError(msg)
        if type(i) == int:
            i = [i]
        if type(i) in [list, slice]:
            idxs = i
        else:
            raise TypeError("Incorrect index type for pst.Features.")
        # Retrieve features
        feats = self.features[idxs, :, k]
        feats = self._format_out(feats)
        return feats

    def _format_aggfeatures(self, aggfeatures, names):
        "Formatter for aggfeatures."
        if len(aggfeatures.shape) == 1:
            self._k_reindices = 1
            aggfeatures = aggfeatures.reshape((len(aggfeatures), 1, 1))
            self.features = aggfeatures
        elif len(aggfeatures.shape) == 2:
            self._k_reindices = 1
            aggfeatures = aggfeatures.reshape((len(aggfeatures),
                                              aggfeatures.shape[1], 1))
        elif len(aggfeatures.shape) == 3:
            self._k_reindices = aggfeatures.shape[2]
            self.features = aggfeatures
        elif len(aggfeatures.shape) > 3:
            raise IndexError("Aggfeatures with more than 3 dimensions.")
        nfeats = self.features.shape[1]
        self.variables = names if names else list(range(nfeats))
        self.out_features = self.variables
        if len(self.variables) != self.features.shape[1]:
            raise IndexError("Incorrect length of variables list.")


def checker_sp_descriptor(retriever, features_o):
    pass
