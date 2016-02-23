
"""
Features Objects
----------------
Objects to manage features in order to retrieve them.
This objects are equivalent to the retrievers object. Retrievers object manage
location points which have to be retrieved from a location input and here it is
manage the retrieving of features from a elements retrieved.

"""

import numpy as np
import warnings
warnings.filterwarnings("always")
from pySpatialTools.utils import NonePerturbation, feat_filter_perturbations


class Features:
    """Features object."""
    __name__ = 'pySpatialTools.FeaturesObject'

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
        ## 1. Check indices into the bounds (WARNING)
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

    @property
    def shape(self):
        return (len(self.features), len(self.variables), self.k_perturb+1)

    ################################# Setters #################################
    ###########################################################################
    def set_descriptormodel(self, descriptormodel):
        """Link the descriptormodel and the feature retriever."""
        if self.typefeat == 'implicit':
            self._format_characterizer(descriptormodel.compute_characs,
                                       descriptormodel._out_formatter)
        elif self.typefeat == 'explicit':
            self._format_characterizer(descriptormodel.reducer,
                                       descriptormodel._out_formatter)
        self._format_variables([])
        self._setdescriptor = True

    ################################ Formatters ###############################
    ###########################################################################
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


class ImplicitFeatures(Features):
    """Element features.
    """
    # Type
    typefeat = 'implicit'

    def _initialization(self):
        ## Main attributes
        self.features = None
        self.variables = None
        self.out_features = None
        self._setdescriptor = False
        ## Other attributes
        self._nullvalue = 0
        ## Perturbation
        self._perturbators = [NonePerturbation()]
        self._map_perturb = lambda x: (0, 0)
        self._dim_perturb = [1]
        ## Function to homogenize output respect aggfeatures
        self._characterizer = lambda x, d: x
        self._format_out_k = lambda x, y1, y2, y3: x
        self._out = 'ndarray'

    def __init__(self, features, perturbations=None, names=[], out_features=[],
                 characterizer=None, out_formatter=None):
        self._initialization()
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
                idxs_notnull, new_idxs, feats_k =\
                    self._features_null_saver(idxs, k_p, k_i)
                feats_k[idxs_notnull] =\
                    self._perturbators[k_p].apply2features_ind(self.features,
                                                               new_idxs, k_i)
            feats_k = self._characterizer(feats_k, d)
            feats_k = self._format_out(feats_k)
            feats.append(feats_k)
        if np.all([type(fea) == np.ndarray for fea in feats]):
            if feats:
                feats = np.concatenate(feats, axis=0)
        return feats

    def _features_null_saver(self, idxs, k_p, k_i):
        new_indices = self._perturbators[k_p].apply2indice(idxs, k_i)
        idxs_notnull = [i for i in range(len(new_indices))
                        if not new_indices[i] >= len(self.features)]
        new_idxs = [idxs[i] for i in idxs_notnull]
        feats = self._nullvalue*np.ones((len(idxs), self.features.shape[1]))
        return idxs_notnull, new_idxs, feats

    ################################# Getters #################################
    ###########################################################################
    def _get_possible_indices(self, idxs=None):
        if idxs is None:
            idxs = slice(0, len(self.features), 1)
        if isinstance(idxs, slice):
            start = 0 if idxs.start is None else idxs.start
            stop = len(self.features)-1 if idxs.stop is None else idxs.stop
            step = 1 if idxs.step is None else idxs.step
            idxs = slice(start, stop, step)
        return idxs

    ################################ Formatters ###############################
    ###########################################################################
    def _format_features(self, features, out_features):
        """Format features."""
        sh = features.shape
        features = features if len(sh) == 2 else features.reshape((sh[0], 1))
        self.features = features
        self.out_features = out_features

    def _format_variables(self, names):
        """Format variables."""
        feats = self[(0, 0), 0]
        if names:
            self.variables = names
            if len(names) != feats.shape[1]:
                msg = "Not matching lengths of variablenames and output feats."
                raise IndexError(msg)
        else:
            if type(feats) == dict:
                self.variables = feats.keys()
            else:
                n_feats = feats.shape[1]
                self.variables = list(range(n_feats))

    ######################### Perturbation management #########################
    ###########################################################################
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

    def add_perturbations(self, perturbations):
        """Add perturbations."""
        perturbations = feat_filter_perturbations(perturbations)
        if type(perturbations) == list:
            for p in perturbations:
                self._dim_perturb.append(p.k_perturb)
                self._create_map_perturbation()
                self._perturbators.append(p)
        else:
            self._dim_perturb.append(perturbations.k_perturb)
            self._create_map_perturbation()
            self._perturbators.append(perturbations)

    def _create_map_perturbation(self):
        """Create the map for getting the perturbation object."""
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


class ExplicitFeatures(Features):
    """Explicit features class. In this class we have explicit representation
    of the features.
    """
    "TODO: adaptation of not only np.ndarray format"
    ## Type
    typefeat = 'explicit'

    def _initialization(self):
        ## Main attributes
        self.features = None
        self.variables = None
        self.out_features = None
        self._setdescriptor = False
        ## Other attributes
        self._nullvalue = 0
        ## Perturbation
        self._perturbators = [NonePerturbation()]
        self._map_perturb = lambda x: (0, 0)
        self._dim_perturb = [1]
        ## Function to homogenize output respect aggfeatures
        self._characterizer = lambda x, d: x
        self._format_out_k = lambda x, y1, y2, y3: x
        self._out = 'ndarray'
        self.possible_regions = None
        self.indices = []
        self._out = 'ndarray'

    def __init__(self, aggfeatures, names=[], nullvalue=None, indices=None,
                 characterizer=None, out_formatter=None):
        self._initialization()
        self._format_aggfeatures(aggfeatures, names, indices)
        self._nullvalue = self._nullvalue if nullvalue is None else nullvalue
        self._format_characterizer(characterizer, out_formatter)

    def _retrieve_feats(self, idxs, c_k, d):
        """Retrieve and prepare output of the features."""
        ## 0. Variable needed
        if type(idxs) == slice:
            idxs = list(range(idxs.start, idxs.stop, idxs.step))
        c_k = [c_k] if type(c_k) == int else c_k
        sh = self.shape[1]
        ## 2. Compute the whole feats
        feats = []
        for i in xrange(len(idxs)):
            new_idxs = list(np.where(self.indices == idxs[i])[0])
            if new_idxs != []:
                feats.append(self.features[new_idxs][:, :, c_k])
            else:
                feats.append(np.ones((1, sh, len(c_k))) * self._nullvalue)
                if self.possible_regions is not None:
                    if new_idxs[0] not in self.possible_regions:
                        raise Exception("Incorrect region selected.")

        feats = np.concatenate(feats, axis=0)
        feats = self._characterizer(feats, d)
        feats = self._format_out(feats)
        return feats

    ################################# Getters #################################
    ###########################################################################
    def _get_possible_indices(self, idxs=None):
        if idxs is None:
            idxs = slice(0, len(self.features), 1)
        if isinstance(idxs, slice):
            start = 0 if idxs.start is None else idxs.start
            stop = len(self.features)-1 if idxs.stop is None else idxs.stop
            step = 1 if idxs.step is None else idxs.step
            idxs = slice(start, stop, step)
        return idxs

    ################################ Formatters ###############################
    ###########################################################################
    def _format_aggfeatures(self, aggfeatures, names, indices):
        """Formatter for aggfeatures."""
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

    ######################### Perturbation management #########################
    ###########################################################################
    def add_perturbations(self, perturbations):
        """Add perturbations."""
        msg = "Aggregated features can not be perturbated."
        msg += "Change order of aggregation."
        raise Exception(msg)


def checker_sp_descriptor(retriever, features_o):
    pass
