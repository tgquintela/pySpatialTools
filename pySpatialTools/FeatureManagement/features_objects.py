
"""
Features Objects
----------------
Objects to manage features in order to retrieve them.
This objects are equivalent to the retrievers object. Retrievers object manage
location points which have to be retrieved from a location input and here it is
manage the retrieving of features from a elements retrieved.


TODO:
-----
Explicit data

"""

import numpy as np
import warnings
warnings.filterwarnings("always")
from pySpatialTools.utils import NonePerturbation, feat_filter_perturbations
from pySpatialTools.utils.util_classes import Neighs_Info


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
        empty = False
        if type(key) == tuple:
            if type(key[0]).__name__ == 'instance':
                empty = key[0].empty()
                i, d, k, _ = key[0].get_information(key[1])
            else:
                neighs_info = Neighs_Info()
                neighs_info.set_information(self.k_perturb, len(self.features))
                neighs_info.set(key)
                empty = neighs_info.empty()
                i, d, k, _ = neighs_info.get_information()
        elif type(key) == int:
            kn = self.k_perturb+1
            i, k, d = np.array([[[key]]]*kn), range(kn), [[None]]*kn
        elif type(key) == list:
            ## Assumption only deep=1
            kn = self.k_perturb+1
            i, k, d = np.array([[key]]*kn), range(kn), [[None]]*kn
        elif type(key) == slice:
            neighs_info = Neighs_Info()
            neighs_info.set_information(self.k_perturb, len(self.features))
            neighs_info.set(key)
            i, d, k, _ = neighs_info.get_information()
        else:
            i, d, k, _ = key.get_information()
            empty = key.empty()
        ## 1. Check indices into the bounds
        if empty:
            return np.ones((1, 1, len(self.variables))) * self._nullvalue
        if type(i) == int:
            if i < 0 or i >= len(self.features):
                raise IndexError("Index out of bounds.")
            i = [[[i]]]
        elif type(i) in [np.ndarray, list]:
            ## Empty escape
            for j_k in range(len(i)):
                for j_i in range(len(i[j_k])):
                    if len(i[j_k][j_i]) == 0:
                        continue
                    bool_overbound = np.max(i[j_k][j_i]) >= len(self.features)
                    bool_lowerbound = np.min(i[j_k][j_i]) < 0
                    if bool_lowerbound or bool_overbound:
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
        print ' '*50, i, k, d, type(i), type(k)
        feats = self._retrieve_feats(i, k, d)
        return feats

    @property
    def shape(self):
        return (len(self.features), len(self.variables), self.k_perturb+1)

    def _retrieve_feats(self, idxs, c_k, d):
        """Retrieve and prepare output of the features.

        Parameters
        ----------
        idxs: list of list of lists, or 3d np.ndarray [ks, iss, nei]
            Indices we want to get features.
        c_k: list
            the different ks we want to get features.
        d: list of list of lists or None [ks, iss, nei, dim]
            the information of relative position we are going to use in the
            characterizer.
        """
        feats = []
        for k in c_k:
            ## Interaction with the features stored and computing characs
            feats_k = self._get_characs_k(k, idxs, d)

            ## Testing #############
            assert(len(feats_k) == len(idxs[k]))
            if type(feats_k) == list:
                assert(type(feats_k[0]) == dict)
            else:
                print feats_k.shape, idxs.shape, idxs
                assert(len(feats_k.shape) == 2)
            ########################

            ## Adding to global result
            feats.append(feats_k)

        if self._out == 'ndarray':
            feats = np.array(feats)
#        if np.all([type(fea) == np.ndarray for fea in feats]):
#            if feats:
#                feats = np.array(feats)
        ## Testing #############
        assert(len(feats) == len(c_k))
        if type(feats) == list:
            pass
        else:
            print feats.shape, idxs.shape, idxs
            assert(len(feats.shape) == 3)
        ########################
        return feats

    ################################# Setters #################################
    ###########################################################################
    def set_descriptormodel(self, descriptormodel, featuresnames=[]):
        """Link the descriptormodel and the feature retriever."""
        descriptormodel.set_functions(type(self.features).__name__, self._out)
        if self.typefeat == 'implicit':
            self._format_characterizer(descriptormodel.compute_characs,
                                       descriptormodel._out_formatter)
        elif self.typefeat == 'explicit':
            self._format_characterizer(descriptormodel.reducer,
                                       descriptormodel._out_formatter)
        self._format_variables(featuresnames)
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
            self[([0], [0.]), 0]
            try:
                self[([0], [0.]), 0]
            except:
                raise TypeError("Incorrect characterizer.")

    def _format_feat_interactors(self, typeidxs):
        """Programming this class in order to fit properly to the inputs and
        the function is going to carry out."""
        if type(self.features) == list:
            msg = "Not adapted to non-array element features."
            raise NotImplementedError(msg)
        elif type(self.features) == np.ndarray:
            if typeidxs is None:
                self._get_characs_k = self._get_characs_k
                self._get_real_data = self._real_data_general
                self._get_virtual_data = self._virtual_data_general
            elif typeidxs == np.ndarray:
                self._get_characs_k = self._get_characs_k
                self._get_real_data = self._real_data_array_array
                self._get_virtual_data = self._virtual_data_array_array
            elif typeidxs == list:
                self._get_characs_k = self._get_characs_k
                self._get_real_data = self._real_data_array_list
                self._get_virtual_data = self._virtual_data_array_list

    ############################ General interaction ##########################
    ###########################################################################
    def _real_data_general(self, idxs, k, k_i=0, k_p=0):
        """General interaction with the real data."""
        print type(self.features), type(idxs), idxs
        if type(self.features) == list:
            if type(idxs) == list:
                feats_k = self._real_data_dict_dict(idxs, k, k_i, k_p)
            elif type(idxs) == np.ndarray:
                feats_k = self._real_data_dict_array(idxs, k, k_i, k_p)
        elif type(self.features) == np.ndarray:
            if type(idxs) == list:
                feats_k = self._real_data_array_list(idxs, k, k_i, k_p)
            elif type(idxs) == np.ndarray:
                feats_k = self._real_data_array_array(idxs, k, k_i, k_p)
        return feats_k

    def _virtual_data_general(self, idxs, k, k_i, k_p):
        """General interaction with the virtual data generated throught
        perturbations."""
        if type(self.features) == list:
            if type(idxs) == list:
                feats_k = self._virtual_data_dict_dict(idxs, k, k_i, k_p)
            elif type(idxs) == np.ndarray:
                feats_k = self._virtual_data_dict_array(idxs, k, k_i, k_p)
        elif type(self.features) == np.ndarray:
            if type(idxs) == list:
                feats_k = self._virtual_data_array_list(idxs, k, k_i, k_p)
            elif type(idxs) == np.ndarray:
                feats_k = self._virtual_data_array_array(idxs, k, k_i, k_p)
        return feats_k


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
        # Reduction of dimensionality (dummy getting first neigh feats)
        self._characterizer = lambda x, d: np.array([e[0] for e in x])
        self._format_out_k = lambda x, y1, y2, y3: x
        self._out = 'ndarray'
        ## Default mutable functions
        self._get_characs_k = self._get_characs_k
        self._get_real_data = self._real_data_general
        self._get_virtual_data = self._virtual_data_general

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

    ############################### Interaction ###############################
    ###########################################################################
    ################################ Candidates ###############################
    def _real_data_array_array(self, idxs, k, k_i=0, k_p=0):
        """Real data array.
        * idxs: (ks, iss_i, nei)
        * feats_k: (iss_i, nei, features)
        """
        feats_k = self.features[idxs[k]]
        assert(len(feats_k.shape) == 3)
        return feats_k

    def _virtual_data_array_array(self, idxs, k, k_i, k_p):
        """Virtual data array.
        * idxs: (ks, iss_i, nei)
        * feats_k: (iss_i, nei, features)
        """
        nfeats = self.features.shape[1]
        sh = idxs.shape
        print idxs, sh
        idxs_k = idxs[k]
        # Compute new indices by perturbating them
        new_idxs = self._perturbators[k_p].apply2indice(idxs_k, k_i)
        # Get rid of the non correct indices
        yes_idxs = np.logical_and(new_idxs >= 0,
                                  new_idxs < len(self.features))
        # Features retrieved from the data stored
        feats_k = np.ones((len(new_idxs), sh[2], nfeats))*self._nullvalue
        feats_k[yes_idxs] =\
            self._perturbators[k_p].apply2features_ind(self.features,
                                                       new_idxs, k_i)
        # Reshaping
        feats_k = feats_k.reshape((sh[1], sh[2], nfeats))
        return feats_k

    def _real_data_array_list(self, idxs, k, k_i=0, k_p=0):
        """
        * idxs: [ks][iss_i][nei]
        * feats_k: [iss_i](nei, features)
        """
        feats_k = []
        for i in range(len(idxs[k])):
            feats_k.append(self.features[idxs[k][i]])
            assert(len(self.features[idxs[k][i]].shape) == 2)
        return feats_k

    def _virtual_data_array_list(self, idxs, k, k_i, k_p):
        """
        * idxs: [ks][iss_i][nei]
        * feats_k: [iss_i](nei, features)
        """
        feats_k = []
        for i in range(len(idxs[k])):
            # Perturbation indices
            new_idxs = self._perturbators[k_p].apply2indice(idxs[k][i], k_i)
            # Indices in bounds
            yes_idxs = [j for j in range(len(new_idxs))
                        if new_idxs[j] < len(self.features)]
            # Features
            feats_ki = np.ones((len(new_idxs), self.features.shape[1]))
            feats_ki = feats_ki*self._nullvalue
            feats_ki[yes_idxs] = self._perturbators[k_p].\
                apply2features_ind(self.features, new_idxs, k_i)
            assert(len(feats_ki.shape) == 2)
            feats_k.append(feats_ki)
        return feats_ki

    def _virtual_data_dict_array(self, idxs, k, k_i, k_p):
        raise NotImplementedError("Not adapted to non-array element features.")

    def _virtual_data_dict_dict(self, idxs, k, k_i, k_p):
        raise NotImplementedError("Not adapted to non-array element features.")

    def _real_data_dict_array(self, idxs, k, k_i=0, k_p=0):
        raise NotImplementedError("Not adapted to non-array element features.")

    def _real_data_dict_dict(self, idxs, k, k_i=0, k_p=0):
        raise NotImplementedError("Not adapted to non-array element features.")

    def _get_characs_k(self, k, idxs, d):
        """Getting characs with array idxs."""
        ## Interaction with the features stored
        print 'characs_inputs', k, idxs, d
        feats_k = self._get_feats_k(k, idxs)
        ## Computing characterizers
        print 'characterizer_inputs', feats_k, d[k], self._out
        feats_k = self._characterizer(feats_k, d[k])
        ## Formatting result
        print feats_k, self._characterizer
        feats_k = self._format_out(feats_k)
        print feats_k
        if type(feats_k) == list:
            pass
        else:
            assert(len(feats_k.shape) == 2)
        return feats_k

    def _get_feats_k(self, k, idxs):
        """Interaction with the stored features."""
        ## Applying k map for perturbations
        k_p, k_i = self._map_perturb(k)
        ## Not perturbed k
        if k_p == 0:
            feats_k = self._get_real_data(idxs, k)
        ## Virtual perturbed data
        else:
            feats_k = self._get_virtual_data(idxs, k, k_i, k_p)
        return feats_k

    ###########################################################################

    ################################# Getters #################################
    ###########################################################################

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
        feats = self[([0], [0]), 0]
        print feats
        if names:
            self.variables = names
        else:
            if type(feats) == list:
                self.variables = feats[0][0].keys()
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
        # Reduction of dimension
        self._characterizer = lambda x, d: np.array([e[0] for e in x])
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

#    def _retrieve_feats_array(self, idxs, c_k, d):
#        """Retrieve and prepare output of the features.
#
#        Parameters
#        ----------
#        idxs: list of list of lists, or 3d np.ndarray
#            Indices we want to get features.
#        c_k: list
#            the different ks we want to get features.
#        d: list of list of lists or None
#            the information of relative position we are going to use in the
#            characterizer.
#
#        TODO
#        ----
#        Use of self.indices
#
#        """
#        ## 0. Variable needed
#        c_k = [c_k] if type(c_k) == int else c_k
#        sh = self.shape[1]
#        ## 2. Compute the whole feats
#        feats = []
#
#
#
#
#        nfeats = self.features.shape[1]
#        sh = idxs.shape
#        feats = []
#        for k in c_k:
#            ## Applying k map for perturbations
#            k_p, k_i = self._map_perturb(k)
#            ## Not perturbed k
#            if k_p == 0:
#                feats_k = self.features[idxs[:, :, k], :, c_k]
#
#
#
#        for i in xrange(len(idxs)):
#            new_idxs = list(np.where(self.indices == idxs[i])[0])
#            if new_idxs != []:
#                feats.append(self.features[new_idxs][:, :, c_k])
#            else:
#                feats.append(np.ones((1, sh, len(c_k))) * self._nullvalue)
#                if self.possible_regions is not None:
#                    if new_idxs[0] not in self.possible_regions:
#                        raise Exception("Incorrect region selected.")
#
#        feats = np.concatenate(feats, axis=0)
#        feats = self._characterizer(feats, d)
#        feats = self._format_out(feats)
#        return feats

    ################################ Candidates ###############################
#    def _real_data_array_array(self, idxs, k, k_i=0, k_p=0):
#        ## TODO: WARNING: Test
#        feats_k = self.features[idxs[:, :, k]]
#        return feats_k

#    def _virtual_data_array_array(self, idxs, k, k_i, k_p):
#        """Virtual data array."""
#        nfeats = self.features.shape[1]
#        sh = idxs.shape
#        idxs_k = idxs[:, :, k].ravel()
#        new_idxs = self._perturbators[k_p].apply2indice(idxs_k, k_i)
#        yes_idxs = np.logical_and(new_idxs >= 0,
#                                  new_idxs < len(self.features))
#        feats_k = np.ones((len(new_idxs), nfeats))*self._nullvalue
#        feats_k[yes_idxs] = self._perturbators[k_p].\
#            apply2features_ind(self.features[:, :, k_i], new_idxs, k_i)
#        #### WARNING: with this reshape
#        feats_k = feats_k.reshape((sh[0], sh[1], feats_k.shape[1]))
#        return feats_k

#    def _real_data_array_list(self, idxs, k, k_i=0, k_p=0):
#        feats_ki = self.features[idxs[k]]
#        return feats_ki

#    def _virtual_data_array_dict(self, idxs, k, k_i, k_p):
#        new_idxs = self._perturbators[k_p].apply2indice(idxs[k], k_i)
#        yes_idxs = [j for j in range(len(new_idxs))
#                    if new_idxs[j] < len(self.features)]
#        feats_ki = np.ones((len(new_idxs), self.features.shape[1]))
#        feats_ki = feats_ki*self._nullvalue
#        feats_ki[yes_idxs] = self._perturbators[k_p].\
#            apply2features_ind(self.features, new_idxs, k_i)
#        return feats_ki

#    def _virtual_data_dict_array(self, idxs, k, k_i, k_p):
#        raise NotImplementedError("Not adapted to non-array element features.")
#
#    def _virtual_data_dict_dict(self, idxs, k, k_i, k_p):
#        raise NotImplementedError("Not adapted to non-array element features.")
#
#    def _real_data_dict_array(self, idxs, k, k_i=0, k_p=0):
#        raise NotImplementedError("Not adapted to non-array element features.")
#
#    def _real_data_dict_dict(self, idxs, k, k_i=0, k_p=0):
#        raise NotImplementedError("Not adapted to non-array element features.")

#    def _get_feats_k_array(self, k, idxs):
#        """Interaction with the stored features."""
#        ## Applying k map for perturbations
#        k_p, k_i = self._map_perturb(k)
#        ## Not perturbed k
#        if k_p == 0:
#            feats_k = self._real_data_array(idxs, k)
#        ## Virtual perturbed data
#        else:
#            feats_k = self._virtual_data_array(idxs, k, k_i, k_p)
#        return feats_k

    def _get_characs_k_array(self, k, idxs, d):
        """Getting characs with array idxs."""
        ## Interaction with the features stored
        feats_k = self._get_feats_k_array(k, idxs)
        ## Computing characterizers
        feats_k = self._characterizer(feats_k, d[k])
        ## Formatting result
        feats_k = self._format_out(feats_k)
        return feats_k

    def _get_characs_k_list(self, k, idxs, d):
        """Interaction with features and computing characs."""
        n_idxs = len(idxs)
        k_p, k_i = self._map_perturb(k)
        feats_k = []
        ## Not perturbed k
        if k_p == 0:
            for i in range(n_idxs):
                feats_ki = self._virtual_data_array_dict(idxs[i], k)
                feats_ki = self._characterizer(feats_ki, d[k][i])
                feats_ki = self._format_out(feats_ki)
            feats_k.append(feats_ki)
        ## Virtual perturbed data
        else:
            for i in range(n_idxs):
                feats_ki = self._virtual_data_array_dict(idxs[i], k, k_i, k_p)
                feats_ki = self._characterizer(feats_ki, d[k][i])
                feats_ki = self._format_out(feats_ki)
            feats_k.append(feats_ki)
        return feats_k

    ###########################################################################

    ################################# Getters #################################
    ###########################################################################

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
