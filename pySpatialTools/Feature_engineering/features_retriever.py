
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
- Support for dictionary

"""

import numpy as np
import warnings
warnings.filterwarnings("always")


class FeaturesRetriever:
    "Method for retrieving features."

    features = []
    featuresnames = []
    #_k_reindices = 1
    k_perturb = 0
    _variables = {}
    _maps_input = None
    _maps_output = None
    _maps_vals_i = None
    _out = 'ndarray'  # dict
    __name__ = "pst.FeaturesRetriever"

    def __init__(self, features_objects, maps_input=None, maps_output=None,
                 out=None, maps_vals_i=None):
        out = out if out in ['ndarray', 'dict'] else None
        self._out = self._out if out is None else out
        self._format_features(features_objects)
        self._format_maps(maps_input, maps_output, maps_vals_i)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i_feat):
        if i_feat < 0 or i_feat >= len(self.features):
            raise IndexError("Not correct index for features.")
        return self.features[i_feat]

    def set_map_vals_i(self, _maps_vals_i):
        "Set how it maps each element of the "
        self._maps_vals_i = _maps_vals_i

    def set_descriptormodel(self, descriptormodel):
        "Set descriptor model."
        ## We are assuming feature 0 is the representative one.
        self.featuresnames =\
            descriptormodel._compute_featuresnames(self.features[0].features)
        ## Set out_features
        for i in range(len(self)):
            out_feat = descriptormodel._compute_featuresnames(self.features[i])
            self.features[i].out_features = out_feat
        ## TODO: Check if all are equal
        self.out_features = out_feat
        init_ = np.ones(self.nfeats) * descriptormodel._nullvalue

        def init_features(_out):
            if _out == 'dict':
                return dict(zip(self.featuresnames, init_))
            elif _out == 'ndarray':
                return init_
        self.initialization_features = init_features
        ## Set each one of the features
        for i in range(len(self.features)):
            self.features[i].set_descriptormodel(descriptormodel)

    @property
    def nfeats(self):
        return len(self.featuresnames)

    def _format_features(self, features_objects):
        "Formatter of features."
        ## Check variables
        if type(features_objects) != list:
            features_objects = [features_objects]
        nfeat = len(features_objects)
        k_perturb = [features_objects[i].k_perturb for i in range(nfeat)]
        vars_o = [set(features_objects[i].variables) for i in range(nfeat)]
        k_rei_bool = [k_perturb[i] == k_perturb[0] for i in range(nfeat)]
        ## Check k perturbations
        if not all(k_rei_bool):
            msg = "Not all the feature objects have the same perturbations."
            raise Exception(msg)
        ## Storing variables
        self._variables = vars_o[0]
        self.k_perturb = k_perturb[0]
        self.features = features_objects
        for i in range(nfeat):
            self.features[i]._out = self._out

    def _format_maps(self, maps_input, maps_output, maps_vals_i):
        "Formatter of maps."
        if maps_input is None:
            self._maps_input = [lambda i, k=0: (i, k)]
        else:
            if type(maps_input).__name__ == 'function':
                self._maps_input = [lambda i, k=0: maps_input(i, k)]
            else:
                self._maps_input = [maps_input]
        if maps_output is None:
            self._maps_output = [lambda i, k=0: (i, k)]
        else:
            if type(maps_output).__name__ == 'function':
                self._maps_output = [lambda i, k=0: maps_output(self, i, k)]
            else:
                self._maps_output = [maps_output]
        if maps_vals_i is None:
            self._maps_vals_i = [lambda i, k=0: i]
        else:
            if type(maps_vals_i).__name__ == 'function':
                self._maps_vals_i = [lambda i, k=0: i]
            else:
                self._maps_vals_i = [maps_vals_i]

    def _get_input_features(self, i, k, typefeats):
        "Get input features."
        ## Retrieve features
        if type(i) == tuple:
            i_input, k_input = self._maps_input[typefeats[0]](i[0], k)
            i_input = i_input, i[1]
        else:
            i_input, k_input = self._maps_input[typefeats[0]](i, k)
        feats_i = self.features[typefeats[1]][i_input, k_input]
        return feats_i

    def _get_output_features(self, idxs, k, typefeats):
        "Get output features."
        ## Retrieve features
        if type(idxs) == tuple:
            idxs_input, k_input = self._maps_output[typefeats[0]](idxs[0], k)
            idxs_input = idxs_input, idxs[1]
        else:
            idxs_input, k_input = self._maps_output[typefeats[0]](idxs, k)
        feats_idxs = self.features[typefeats[1]][idxs_input, k_input]
        return feats_idxs

    def _get_prefeatures(self, i, neighs_info, k, typefeats):
        """General interaction with features object to get point features from
        it.
        """
        ## 0. Prepare list of k
        ks = range(self.k_perturb+1) if k is None else k
        ks = [ks] if type(ks) == int else ks
        ## 1. Loop over possible ks and compute descriptors
        t_feat_in, t_feat_out = typefeats[0:2], typefeats[2:4]
        desc_i = self._get_input_features(i, ks, t_feat_in)
        desc_neigh = self._get_output_features(neighs_info, ks, t_feat_out)
        return desc_i, desc_neigh

    def _get_vals_i(self, i, k, typefeats):
        "Get how to store the final result."
        ## 0. Prepare variable needed
        vals_i = []
        ks = list(range(self.k_perturb+1)) if k is None else k
        ks = [ks] if type(ks) == int else ks
        ## 1. Loop over possible ks and compute vals_i
        for k in ks:
            vals_i.append(self._maps_vals_i[typefeats[4]](i, k))
        return vals_i


class Features:
    "Features object."

    _out = 'ndarray'
    __name__ = 'pst.FeaturesObject'
    _setdescriptor = False

    def __len__(self):
        return len(self.features)

    def __getitem__(self, key):
        """Possible ways to get items in pst.Features classes:
        * (i, k)
        * (neighs, k)
        * (neighs_info, k)
            where neighs_info is a tuple which could contain (neighs, dists) or
            (neighs,)
        """
        ## 0. Format inputs
        if type(key) == int:
            i, k, d = [key], range(self.k_perturb+1), None
        if type(key) == list:
            i, k, d = key, range(self.k_perturb+1), None
        if type(key) == tuple:
            assert len(key) == 2
            if type(key[0]) == tuple:
                if len(key[0]) == 2:
                    i, k, d = key[0][0], key[1], key[0][1]
                else:
                    i, k, d = key[0][0], key[1], None
            else:
                if type(key[0]) == int:
                    i = [key[0]]
                    k = key[1]
                else:
                    i = [key[0]] if type(key[0]) == int else key[0]
                    i = list(i) if type(i) == np.ndarray else i
                    assert type(i) in [list, slice]
                    if type(i) == list:
                        n_len_i = len(i)
                    else:
                        i = self._get_possible_indices(i)
                        n_len_i = len(range(i.start, i.stop, i.step))
                    msg = "Ambiguous input in __getitem__ of pst.Features."
                    warnings.warn(msg, SyntaxWarning)
                    if type(key[1]) in [slice, int]:
                        d = None
                        k = [key[1]] if type(key[1]) == int else key[1]
                    else:
                        # Assumption of list or np.ndarray
                        types = [type(j) == int for j in key[1]]
                        if len(key[1]) == n_len_i:
                            d = None
                            if np.all(types):
                                k = list(key[1])
                            else:
                                k = range(self.k_perturb+1)
                                d = [float(j) for j in key[1]]
                        else:
                            msg = "Too ambiguous..."
                            msg += " Dangerous casting to integers is done."
                            warnings.warn(msg, SyntaxWarning)
                            k = [int(j) for j in key[1]]
                            d = None
        # If the input is with neighs_info
        if type(i) == tuple:
            i, d = i
        else:
            d = None
        # Slice input
        if isinstance(i, slice):
            i = self._get_possible_indices(i)
        if isinstance(k, slice):
            start = 0 if k.start is None else k.start
            stop = self.k_perturb+1 if k.stop is None else k.stop
            step = 1 if k.step is None else k.step
            k = range(start, stop, step)
        ## 1. Check indices into the bounds
        if type(i) == int:
            if i < 0 or i >= len(self.features):
                raise IndexError("Index out of bounds.")
            i = [i]
        elif type(i) in [np.ndarray, list]:
            if np.min(i) < 0 or np.max(i) >= len(self.features):
                raise IndexError("Indices out of bounds.")
        elif type(i) == slice:
            if i.start < 0 or i.stop >= len(self.features):
                raise IndexError("Indices out of bounds.")
        ## 2. Format k
        if k is None:
            k = list(range(self.k_perturb+1))
        else:
            if type(k) == int:
                k = [k]
            elif type(k) in [np.ndarray, list]:
                k = list(k)
                if np.min(k) < 0 or np.max(k) >= (self.k_perturb+1):
                    msg = "Index of k perturbation is out of bounds."
                    raise IndexError(msg)
        if type(k) != list:
            raise TypeError("Incorrect type of k perturbation index.")
        # Retrive features
        feats = self._retrieve_feats(i, k, d)
        return feats

    def set_descriptormodel(self, descriptormodel):
        "Link the descriptormodel and the feature retriever."
        if self._type == 'point':
            self._format_characterizer(descriptormodel.compute_characs,
                                       descriptormodel._out_formatter)
        elif self._type == 'aggregated':
            self._format_characterizer(descriptormodel.reducer,
                                       descriptormodel._out_formatter)
        self._format_variables([])
        self._setdescriptor = True

    @property
    def shape(self):
        return (len(self.features), len(self.variables), self.k_perturb+1)

    def _format_out(self, feats):
        "Transformation array-dict."
        feats_o = self._format_out_k(feats, self.out_features, self._out,
                                     self._nullvalue)
        return feats_o

    def _format_characterizer(self, characterizer, out_formatter):
        """Format characterizer function. It is needed to homogenize outputs in
        order to have the same output type as the aggfeatures.
        """
        if characterizer is not None:
            self._characterizer = characterizer
        if out_formatter is not None:
            self._format_out_k = out_formatter
        if not (characterizer is None or out_formatter is None):
            self[(0, 0.), 0]
            try:
                self[(0, 0), 0]
            except:
                raise TypeError("Incorrect characterizer.")


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
    ## Function to homogenize output respect aggfeatures
    _characterizer = lambda s, x, d: x
    _format_out_k = lambda s, x, y1, y2, y3: x
    # Type
    _type = 'point'
    ## Perturbation
    _perturbators = [[]]
    _map_perturb = lambda s, x: (0, 0)
    _dim_perturb = []
    k_perturb = 0

    def __init__(self, features, perturbations=None, names=[], out_features=[],
                 characterizer=None, out_formatter=None):
        self._format_features(features, out_features)
        self._format_characterizer(characterizer, out_formatter)
        self._format_variables(names)
        self._format_perturbation(perturbations)

    @property
    def k_perturb(self):
        if self._dim_perturb:
            return np.sum(self._dim_perturb)-1

    def _retrieve_feats(self, idxs, c_k, d):
        "Retrieve and prepare output of the features."
        feats = []
        for k in c_k:
            k_p, k_i = self._map_perturb(k)
            if k_p == 0:
                feats_k = self.features[idxs]
            else:
                feats_k =\
                    self._perturbators[k_p].apply_ind(self.features, idxs, k_i)
            print '00', feats_k
            feats_k = self._characterizer(feats_k, d)
            print '01', feats_k
            feats_k = self._format_out(feats_k)
            print '02', feats_k
            feats.append(feats_k)
        if np.all([type(fea) == np.ndarray for fea in feats]):
            if feats:
                feats = np.concatenate(feats, axis=0)
        return feats

    def _get_possible_indices(self, idxs=None):
        if idxs is None:
            idxs = slice(0, len(self.features), 1)
        if isinstance(idxs, slice):
            start = 0 if idxs.start is None else idxs.start
            stop = len(self.features)-1 if idxs.stop is None else idxs.stop
            step = 1 if idxs.step is None else idxs.step
            idxs = slice(start, stop, step)
        return idxs

    def _format_features(self, features, out_features):
        "Format features."
        sh = features.shape
        features = features if len(sh) == 2 else features.reshape((sh[0], 1))
        self.features = features
        self.out_features = out_features

    def _format_variables(self, names):
        "Format variables."
        feats = self[(0, 0), 0]
        print feats
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

    def _format_perturbation(self, perturbations):
        "Format initial perturbations."
        if perturbations is None:
            def _map_perturb(x):
                if x != 0:
                    raise IndexError("Not perturbation available.")
                return 0, 0
            self._map_perturb = _map_perturb
            self._dim_perturb = [1]
        else:
            self.add_perturbations(perturbations)

    def add_perturbations(self, perturbations):
        "Add perturbations."
        if type(perturbations) == list:
            for p in perturbations:
                self._dim_perturb.append(p.k_perturb)
                self.k_perturb = np.sum(self._dim_perturb)-1
                self._create_map_perturbation()
                self._perturbators.append(p)
        else:
            self._dim_perturb.append(perturbations.k_perturb)
            self._create_map_perturbation()
            self._perturbators.append(perturbations)

    def _create_map_perturbation(self):
        "Create the map for getting the perturbation object."
        ## 0. Creation of the mapper array
        limits = np.cumsum([0] + list(self._dim_perturb))
        sl = [slice(limits[i], limits[i+1]) for i in range(len(limits)-1)]
        ## Build a mapper
        mapper = np.zeros((np.sum(self._dim_perturb), 2)).astype(int)
        for i in range(len(sl)):
            inds = np.zeros((sl[i].stop-sl[i].start, 2))
            inds[:, 0] = i
            inds[:, 1] = np.arange(sl[i].stop-sl[i].start)
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

    def add_aggregations(self, discretization_info, regret, agg_funct):
        "Create aggregation of fetures to favour the computation."
        ## 0. Preparing the inputs
        if type(discretization_info) == tuple:
            locs, discretizor = discretization_info
            regs = discretizor.discretize(locs)
        else:
            regs = discretization_info
        u_regs = np.unique(regs)
        u_regs = u_regs.reshape((len(u_regs), 1))

        ## 1. Compute aggregation
        sh = self.shape
        agg = np.ones((len(u_regs), sh[1], sh[2])) * self._nullvalue
        for i in xrange(len(u_regs)):
            neighs_info = regret.retrieve_neighs(u_regs[i])
            if list(neighs_info[0]) != []:
                for k in range(self.k_perturb+1):
                    agg[i, :, k] = agg_funct(self[neighs_info, k],
                                             neighs_info[1])
            else:
                sh = self.shape
                agg[i, :, :] = np.ones((sh[1], sh[2])) * self._nullvalue

        ## 2. Prepare output
        agg = AggFeatures(agg, indices=u_regs,
                          characterizer=self._characterizer)

        return agg


class AggFeatures(Features):
    "Aggregate features class."
    "TODO: adaptation of not only np.ndarray format"

    ## Main attributes
    features = None
    variables = None
    out_features = None
    _characterizer = None
    ## Other attributes
    _nullvalue = 0
    possible_regions = None
    k_perturb = 0
    indices = []
    ## Type
    _type = 'aggregated'

    def __init__(self, aggfeatures, names=[], nullvalue=None, indices=None,
                 characterizer=None, out_formatter=None):
        self._format_aggfeatures(aggfeatures, names, indices)
        self._nullvalue = self._nullvalue if nullvalue is None else nullvalue
        self._format_characterizer(characterizer, out_formatter)

    def _retrieve_feats(self, idxs, c_k, d):
        "Retrieve and prepare output of the features."
        ## 0. Variable needed
        if type(idxs) == slice:
            idxs = list(range(idxs.start, idxs.stop, idxs.step))
        c_k = [c_k] if type(c_k) == int else c_k
        sh = self.shape[1]
        ## 2. Compute the whole feats
        feats = []
        for i in xrange(len(idxs)):
            new_idxs = list(np.where(self.indices == idxs[i])[0])
            print new_idxs
            if new_idxs != []:
                feats.append(self.features[new_idxs][:, :, c_k])
            else:
                feats.append(np.ones((1, sh, len(c_k))) * self._nullvalue)
                if self.possible_regions is not None:
                    if new_idxs[0] not in self.possible_regions:
                        raise Exception("Incorrect region selected.")
##          Ensure feats dim
#            if hasattr(idxs, "__len__"):
#                length = len(idxs)
#            else:
#                length = len(range(idxs.start, idxs.stop, idxs.step))
#            if type(feats_k) == np.ndarray:
#                feats_k = feats_k.reshape((length, feats_k.shape[1], 1))
#                feats.append(feats_k)

        feats = np.concatenate(feats, axis=0)
        feats = self._format_out(self._characterizer(feats, d))
        return feats

    def _get_possible_indices(self, idxs=None):
        if idxs is None:
            idxs = slice(0, len(self.features), 1)
        if isinstance(idxs, slice):
            start = 0 if idxs.start is None else idxs.start
            stop = len(self.features)-1 if idxs.stop is None else idxs.stop
            step = 1 if idxs.step is None else idxs.step
            idxs = slice(start, stop, step)
        return idxs

    def _format_aggfeatures(self, aggfeatures, names, indices):
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
        self._format_variables(names)
        self.indices = indices
        self.k_perturb = aggfeatures.shape[2]-1

    def _format_variables(self, names):
        nfeats = self.features.shape[1]
        self.variables = names if names else list(range(nfeats))
        self.out_features = self.variables
        if len(self.variables) != self.features.shape[1]:
            raise IndexError("Incorrect length of variables list.")

    def add_aggregation(self, aggfeatures, indices):
        self.aggfeatures.append(aggfeatures)
        self.indices.append(indices)


def checker_sp_descriptor(retriever, features_o):
    pass
