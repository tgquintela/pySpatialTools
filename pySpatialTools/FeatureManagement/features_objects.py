
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
from pySpatialTools.utils.perturbations import NonePerturbation,\
    feat_filter_perturbations
from pySpatialTools.utils.neighs_info import Neighs_Info,\
    neighsinfo_features_preformatting_tuple
from Descriptors import DummyDescriptor, DistancesDescriptor


class BaseFeatures:
    """Features object."""
    __name__ = 'pySpatialTools.FeaturesObject'

    def _global_initialization(self):
        ## Main attributes
        self.features = None
        self.variables = []
        self._setdescriptor = False
        ## Other attributes
        self._nullvalue = 0
        ## Perturbation
        self._perturbators = [NonePerturbation()]
        self._map_perturb = lambda x: (0, 0)
        self._dim_perturb = [1]
        ## Descriptormodel and output
        self.descriptormodel = DummyDescriptor()
        self._out = 'ndarray'
        self.out_features = []  # They have to be list of str
#        # Reduction of dimensionality (dummy getting first neigh feats)
#        self._descriptormodel = lambda x, d: np.array([e[0] for e in x])
#        self._format_out_k = lambda x, y1, y2, y3: x

    def __len__(self):
        """Size of the possible input index."""
        if self.shape[0] is None:
            return 0
        return self.shape[0]

    def compute(self, key):
        """The main function. Retrieve the features required with the `key`
        variable and compute the descriptors using the descriptormodel.

        Parameters
        ----------
        key: tuple, int, list, np.ndarray or slice
            the information to get features and compute descriptors associated.
            The standard input form of the key is:
                * (i, k)
                * (neighs, k)
                * (neighs_info, k) where neighs_info is a tuple which could
                    contain (neighs, dists) or (neighs,)

        Returns
        -------
        feats: np.ndarray or list
            the descriptors computed by applying the descriptormodel to the
            features got.

        """
        return self[key]

    def __getitem__(self, key):
        """Possible ways to get items in pst.Features classes:

        Parameters
        ----------
        key: tuple, int, list, np.ndarray or slice
            the information to get features and compute descriptors associated.
            The standard input form of the key is:
                * (i, k)
                * (neighs, k)
                * (neighs_info, k) where neighs_info is a tuple which could
                    contain (neighs, dists) or (neighs,)

        Returns
        -------
        feats: np.ndarray or list
            the descriptors computed by applying the descriptormodel to the
            features got.

        """
        ## 0. Format inputs
        empty = False
        if type(key) == tuple:
            if type(key[0]).__name__ == 'instance':
                empty = key[0].empty()
                i, d, k, _ = key[0].get_information(key[1])
            elif type(key[0]) in [list, np.ndarray]:
                kn = self.k_perturb+1
                i, k, d = neighsinfo_features_preformatting_tuple(key, kn)
                empty = not np.array(i).size
            else:
                # Preformatting
                key_i = [key[0]] if type(key[0]) == int else key[0]
                key = key_i, key[1]
                # Instantiation
                neighs_info = Neighs_Info()
                neighs_info.set_information(self.k_perturb, self.shape[0])
                neighs_info.set(key)
                empty = neighs_info.empty()
                # Get information
                i, d, k, _ = neighs_info.get_information()
        elif type(key) == int:
            kn = self.k_perturb+1
            i, k, d = np.array([[[key]]]*kn), range(kn), [[None]]*kn
        elif type(key) in [list, np.ndarray]:
            ## Assumption only deep=1
            kn = self.k_perturb+1
            key = [[idx] for idx in key]
            i, k, d = np.array([key]*kn), range(kn), [[None]*len(key)]*kn
        elif type(key) == slice:
            len_f, kn = len(self), self.k_perturb+1
            start = 0 if key.start is None else key.start
            stop = len_f if key.stop >= len_f else key.stop
            step = 1 if key.step is None else key.step
            i = [[[j] for j in range(start, stop, step)]]*kn
            d, k = [[None]*len(i[0])]*kn, range(kn)
#            neighs_info = Neighs_Info()
#            neighs_info.set_information(self.k_perturb, self.shape[0])
#            neighs_info.set(key)
#            i, d, k, _ = neighs_info.get_information()
#            print i
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
                    if self.shape[0] is not None:
                        bool_overbound = np.max(i[j_k][j_i]) >= self.shape[0]
                        bool_lowerbound = np.min(i[j_k][j_i]) < 0
#                        print i[j_k][j_i], self.shape[0]
                        if bool_lowerbound or bool_overbound:
#                            print i, k, d, bool_overbound, bool_lowerbound
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
#        print ' '*50, i, k, d, type(i), type(k)
        feats = self._retrieve_feats(i, k, d)
        return feats

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
            descriptormodel.

        Returns
        -------
        feats: list or np.ndarray
            the descriptors computed from the extracted features and using
            the associated descriptormodel.

        """
        feats = []
        for k in c_k:
            ## Interaction with the features stored and computing characs
            feats_k = self._get_characs_k(k, idxs, d)

#            ## Testing #############
#            print feats_k
#            assert(len(feats_k) == len(idxs[k]))
#            if type(feats_k) == list:
#                assert(type(feats_k[0]) == dict)
#            else:
#                print feats_k.shape, idxs
#                assert(len(feats_k.shape) == 2)
#            ########################

            ## Adding to global result
            feats.append(feats_k)
        if self._out == 'ndarray':
            ## TODO: Assert not dict types
            feats = np.array(feats)
#        if np.all([type(fea) == np.ndarray for fea in feats]):
#            if feats:
#                feats = np.array(feats)
#        ## Testing #############
#        assert(len(feats) == len(c_k))
#        if type(feats) == list:
#            pass
#        else:
#            print feats.shape, idxs.shape, idxs
#            assert(len(feats.shape) == 3)
#        ########################
        return feats

    def _descriptormodel(self, idxs, d):
        """Function to compute descriptors from neighs and spatial relative
        positions.

        idxs: list or np.ndarray
            the indices of the neighs in orther to get the associated features.
        d: list or np.ndarray
            the spatial relative positions of the neighbours.

        Returns
        -------
        desc: list or np.ndarray
            the descriptors computed.

        """
        return self.descriptormodel.compute(idxs, d)

    def _format_out(self, feats):
        """Transformation array-dict.

        Parameters
        ----------
        feats: list or np.ndarray
            the descriptors computed after retrieving.

        Returns
        -------
        feats_o: list or np.ndarray
            the features formatted to be output.

        """
        feats_o =\
            self.descriptormodel._out_formatter(feats, self.out_features,
                                                self._out, self._nullvalue)
        return feats_o

    ################################# Setters #################################
    ###########################################################################
    def set_descriptormodel(self, descriptormodel, out_features=[],
                            out_type=None):
        """Link the descriptormodel and the feature retriever.

        Parameters
        ----------
        descriptormodel: pst.BaseDescriptormodel
            the descriptormodel information.
        out_features: list (default=[])
            the output features names.
        out_type: str, optional ['array', 'ndarray', 'dict'] (default=None)
            the output type we want to retrieve descriptors.

        """
        ## Format out
        self._format_outtypes(out_type)
        ## Set descriptormodel
        descriptormodel.set_functions(type(self.features).__name__, self._out)
        self.descriptormodel = descriptormodel
        ## Format outfeatures
        self._format_out_features(out_features)
        ## Test
        out = self[([0], [0.]), 0]
        if type(out[0][0]) == dict and self._out == 'ndarray':
            warnings.warn("Change in output type because incoherence.")
            self._out = 'dict'
        self._setdescriptor = True

    ################################ Formatters ###############################
    ###########################################################################
    def _format_descriptormodel(self, descriptormodel, out_features, out_type):
        """Format descriptormodel.

        Parameters
        ----------
        descriptormodel: pst.BaseDescriptormodel
            the descriptormodel information.
        out_features: list
            the output features names.
        out_type: str, optional ['array', 'ndarray', 'dict']
            the output type we want to retrieve descriptors.

        """
        if descriptormodel is not None:
            self.set_descriptormodel(descriptormodel, out_features, out_type)
        else:
            self.set_descriptormodel(self.descriptormodel, out_features,
                                     out_type)

    def _format_out_features(self, out_features):
        """Format the output features names.

        Parameters
        ----------
        out_features: list
            the output features names.

        """
        ## Set out featurenames
        if out_features:
            self.out_features = out_features
        else:
            self.out_features =\
                self.descriptormodel._f_default_names(self.features)

    def _format_outtypes(self, _out):
        """Format the output types of the descriptors.

        Parameters
        ----------
        out_: str, optional ['array', 'ndarray', 'dict']
            the output type we want to retrieve descriptors.

        """
        if _out is not None:
            assert(_out in ['ndarray', 'array', 'dict'])
            self._out = 'ndarray' if _out in ['ndarray', 'array'] else _out

#    def _format_feat_interactors(self, typeidxs):
#        """Programming this class in order to fit properly to the inputs and
#        the function is going to carry out."""
#        if type(self.features) == list:
#            msg = "Not adapted to non-array element features."
#            raise NotImplementedError(msg)
#        elif type(self.features) == np.ndarray:
#            if typeidxs is None:
#                self._get_characs_k = self._get_characs_k
#                self._get_real_data = self._real_data_general
#                self._get_virtual_data = self._virtual_data_general
#            elif typeidxs == np.ndarray:
#                self._get_characs_k = self._get_characs_k
#                self._get_real_data = self._real_data_array_array
#                self._get_virtual_data = self._virtual_data_array_array
#            elif typeidxs == list:
#                self._get_characs_k = self._get_characs_k
#                self._get_real_data = self._real_data_array_list
#                self._get_virtual_data = self._virtual_data_array_list

    ############################ General interaction ##########################
    ###########################################################################
    def _real_data_general(self, idxs, k, k_i=0, k_p=0):
        """General interaction with the real data.

        Parameters
        ----------
        idxs: list or np.ndarray
            the indices of the neighs in orther to get the associated features.
        k: int
            the global perturbation index.
        k_i: int (default=0)
            the perturbation index of a given perturbation type.
        k_p: int (default=0)
            the perturbation index type index.

        Returns
        -------
        feats_k: list or np.ndarray
            the descriptors of each given `k`.

        """
        if type(self.features) == list:
            if type(idxs) == list:
                feats_k = self._real_data_dict_list(idxs, k, k_i, k_p)
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
        perturbations.

        Parameters
        ----------
        idxs: list or np.ndarray
            the indices of the neighs in orther to get the associated features.
        k: int
            the global perturbation index.
        k_i: int (default=0)
            the perturbation index of a given perturbation type.
        k_p: int (default=0)
            the perturbation index type index.

        Returns
        -------
        feats_k: list or np.ndarray
            the descriptors of each given `k`.

        """
        if type(self.features) == list:
            if type(idxs) == list:
                feats_k = self._virtual_data_dict_list(idxs, k, k_i, k_p)
            elif type(idxs) == np.ndarray:
                feats_k = self._virtual_data_dict_array(idxs, k, k_i, k_p)
        elif type(self.features) == np.ndarray:
            if type(idxs) == list:
                feats_k = self._virtual_data_array_list(idxs, k, k_i, k_p)
            elif type(idxs) == np.ndarray:
                feats_k = self._virtual_data_array_array(idxs, k, k_i, k_p)
        return feats_k

    ############################## Main interaction ###########################
    ###########################################################################
    def _get_characs_k(self, k, idxs, d):
        """Getting characs with array idxs.

        Parameters
        ----------
        k: int
            the global perturbation index.
        idxs: list or np.ndarray
            the indices of the neighs in orther to get the associated features.
        d: list or np.ndarray
            the spatial relative positions of the neighbours.

        Returns
        -------
        feats_k: list or np.ndarray
            the descriptors of each given `k`.

        """
        ## Interaction with the features stored
#        print 'characs_inputs', k, idxs, d
        feats_k = self._get_feats_k(k, idxs)
        d_k = self._get_relpos_k(k, d)
        ## Computing descriptormodels
#        print d
#        print 'descriptormodel_inputs', feats_k, d_k, self._out
        feats_k = self._descriptormodel(feats_k, d_k)
        ## Formatting result
#        print feats_k, self._descriptormodel
        feats_k = self._format_out(feats_k)

#        #### Testing #######################
#        print feats_k, type(feats_k)
#        if type(feats_k) == list:
#            pass
#        else:
#            assert(len(feats_k.shape) == 2)
#        ####################################
        return feats_k

    def _get_relpos_k(self, k, d):
        """Get the relative position of the `k` perturbation.

        Parameters
        ----------
        k: int
            the global perturbation index.
        d: list or np.ndarray
            the spatial relative positions of the neighbours.

        Returns
        -------
        d_k: list or np.ndarray
            the spatial relative positions of the neighbours in the
            perturbation `k`.

        """
        k_p, k_i = self._map_perturb(k)
        return d[k_i]

    def complete_desc_i(self, i, neighs_info, desc_i, desc_neighs, vals_i):
        """Complete measure. Calling descriptormodel. Wrapper to
        descriptormodel.

        Parameters
        ----------
        i: int, list or np.ndarray
            the index information.
        neighs_info:
            the neighbourhood information of each `i`.
        desc_i: np.ndarray or list of dict
            the descriptors associated to the elements `iss`.
        desc_neighs: np.ndarray or list of dict
            the descriptors associated to the neighbourhood elements of each
            `iss`.
        vals_i: list or np.ndarray
            the storable index information.

        Returns
        -------
        descriptors: list
            the descriptors for each element.

        """
        descriptors =\
            self.descriptormodel.complete_desc_i(i, neighs_info, desc_i,
                                                 desc_neighs, vals_i)
        return descriptors


class ImplicitFeatures(BaseFeatures):
    """Element features. Perturbations are implicit and have to be computed
    each time we want them.
    The features will be only with deep=2 [iss]{features} or (iss, features).
    """
    # Type
    typefeat = 'implicit'

    def _initialization(self):
        ## Default mutable functions
        self._get_characs_k = self._get_characs_k
        self._get_real_data = self._real_data_general
        self._get_virtual_data = self._virtual_data_general
        ## Specific class parameters
        self.relabel_indices = None  # TODO
        self.features_array = None

    def __init__(self, features, descriptormodel=None, perturbations=None,
                 out_type='ndarray', names=[], out_features=[]):
        """Features implicitly computed. The perturbations are computed
        on the fly, there are not precomputed.

        Parameters
        ----------
        features: list or np.ndarray
            the features stored. [iss][feats]
        descriptormodel: pst.BaseDescriptormodel (default=None)
            the descriptormodel information.
        perturbations: pst.BasePerturbation (default=None)
            the perturbations applied to that features.
        out_type: str, optional (default='ndarray')
            the output type we want to retrieve descriptors.
        names: list (default=[])
            the stored features names.
        out_features: list (default=[])
            the output features names.

        """
        self._global_initialization()
        self._initialization()
        self._format_outtypes(out_type)
        self._format_features(features)
        self._format_descriptormodel(descriptormodel, out_features, out_type)
        self._format_variables(names, out_features)
        self._format_perturbation(perturbations)

    @property
    def k_perturb(self):
        """Number of perturbations."""
        if self._dim_perturb:
            return np.sum(self._dim_perturb)-1

    @property
    def shape(self):
        """Size information.

        Returns
        -------
        n: int
            the number of instances of the stored features.
        nfeats: int
            the number of features variables stored.
        ks: int
            the number of perturbations plus the unperturbated.

        """
        n = len(self.features)
        nfeats = None if self.variables is None else len(self.variables)
        ks = self.k_perturb+1
        return n, nfeats, ks

    def export_features(self):
        """Export features to be copied or reinstantiated.

        Returns
        -------
        object_feats: class
            the class of the features to be instantiated.
        core_features: np.ndarray or list
            the features stored.
        pars_fea_o_in: dict
            the parameters needed to reinstantiate the same object.

        """
        object_feats = ImplicitFeatures
        core_features = self.features
        pars_fea_o_in = {}
        pert = None if len(self._perturbators) == 1 else self._perturbators[1:]
        pars_fea_o_in['perturbations'] = pert
        pars_fea_o_in['names'] = self.variables
        pars_fea_o_in['descriptormodel'] = self.descriptormodel
        return object_feats, core_features, pars_fea_o_in

    ############################### Interaction ###############################
    ###########################################################################
    ################################ Candidates ###############################
    def _real_data_array_array(self, idxs, k, k_i=0, k_p=0):
        """Real data array.

        Parameters
        ----------
        idxs: list or np.ndarray
            the indices of the neighs in orther to get the associated features.
            The standards are: (ks, iss_i, nei)
        k: int
            the global perturbation index.
        k_i: int (default=0)
            the perturbation index of a given perturbation type.
        k_p: int (default=0)
            the perturbation index type index.

        Returns
        -------
        feats_k: list or np.ndarray
            the descriptors of each given `k`. Standard: (iss_i, nei, features)

        """
        feats_k = self.features[idxs[k]]
        assert(len(feats_k.shape) == 3)
        return feats_k

    def _virtual_data_array_array(self, idxs, k, k_i, k_p):
        """Virtual data array.

        Parameters
        ----------
        idxs: list or np.ndarray
            the indices of the neighs in orther to get the associated features.
            The standards are: (ks, iss_i, nei)
        k: int
            the global perturbation index.
        k_i: int (default=0)
            the perturbation index of a given perturbation type.
        k_p: int (default=0)
            the perturbation index type index.

        Returns
        -------
        feats_k: list or np.ndarray
            the descriptors of each given `k`. Standard: (iss_i, nei, features)

        TODO: Indices-labels support for each iss.
        """
        nfeats = self.features.shape[1]
        sh = idxs.shape
        # Compute new indices by perturbating them
        new_idxs = self._perturbators[k_p].apply2indice(idxs[k], k_i)
        # Get rid of the non correct indices
        yes_idxs = np.logical_and(new_idxs >= 0,
                                  new_idxs < len(self.features))
        # Features retrieved from the data stored
        feats_k = np.ones((len(new_idxs), sh[2], nfeats))*self._nullvalue
        feats_k[yes_idxs] =\
            self._perturbators[k_p].apply2features_ind(self.features,
                                                       new_idxs.ravel(), k_i)
#        # Reshaping
#        feats_k = feats_k.reshape((sh[1], sh[2], nfeats))
        return feats_k

    def _real_data_array_list(self, idxs, k, k_i=0, k_p=0):
        """Real data input idxs as array but output as list array.

        Parameters
        ----------
        idxs: list or np.ndarray
            the indices of the neighs in orther to get the associated features.
            The standards are: [ks][iss_i][nei]
        k: int
            the global perturbation index.
        k_i: int (default=0)
            the perturbation index of a given perturbation type.
        k_p: int (default=0)
            the perturbation index type index.

        Returns
        -------
        feats_k: list or np.ndarray
            the descriptors of each given `k`. Standard: [iss_i](nei, features)

        """
        feats_k = []
        for i in range(len(idxs[k])):
            feats_k.append(self.features[idxs[k][i]])
            assert(len(self.features[idxs[k][i]].shape) == 2)
        return feats_k

    def _virtual_data_array_list(self, idxs, k, k_i, k_p):
        """Virtual data output list of array.

        Parameters
        ----------
        idxs: list or np.ndarray
            the indices of the neighs in orther to get the associated features.
            The standards are: [ks][iss_i][nei]
        k: int
            the global perturbation index.
        k_i: int (default=0)
            the perturbation index of a given perturbation type.
        k_p: int (default=0)
            the perturbation index type index.

        Returns
        -------
        feats_k: list or np.ndarray
            the descriptors of each given `k`. Standard: [iss_i](nei, features)

        WARNiNG: k_i in idxs
        """
        feats_k = []
#        print idxs, k, k_i, k_p
        for i in range(len(idxs[k_i])):
            # Perturbation indices
            new_idxs = self._perturbators[k_p].apply2indice(idxs[k_i][i], k_i)
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
        return feats_k

    def _virtual_data_dict_array(self, idxs, k, k_i, k_p):
        """Virtual data input as array and output as dict.

        Parameters
        ----------
        idxs: list or np.ndarray
            the indices of the neighs in orther to get the associated features.
            The standards are: (ks, iss_i, nei)
        k: int
            the global perturbation index.
        k_i: int (default=0)
            the perturbation index of a given perturbation type.
        k_p: int (default=0)
            the perturbation index type index.

        Returns
        -------
        feats_k: list or np.ndarray
            the descriptors of each given `k`. Standard: [iss_i][nei]{features}

        """
        feats_k = self._virtual_data_dict_list(idxs, k, k_i, k_p)
        return feats_k

    def _virtual_data_dict_list(self, idxs, k, k_i, k_p):
        """Virtual data retrieved by array.

        Parameters
        ----------
        idxs: list or np.ndarray
            the indices of the neighs in orther to get the associated features.
            The standards are: (ks, iss_i, nei)
        k: int
            the global perturbation index.
        k_i: int (default=0)
            the perturbation index of a given perturbation type.
        k_p: int (default=0)
            the perturbation index type index.

        Returns
        -------
        feats_k: list or np.ndarray
            the descriptors of each given `k`. Standard: [iss_i][nei]{features}

        """
        feats_k = []
        for i in range(len(idxs[k_i])):
            # Perturbation indices
            new_idxs = self._perturbators[k_p].apply2indice(idxs[k_i][i], k_i)
            # Indices in bounds
            yes_idxs = [j for j in range(len(new_idxs))
                        if new_idxs[j] < len(self.features[0])]
            # Features
            feats_ki = []
            for j in range(len(new_idxs)):
                if j in yes_idxs:
                    aux_feat = self._perturbators[k_p].\
                        apply2features_ind(self.features, new_idxs[j], k_i)
                    feats_ki.append(aux_feat)
                else:
                    feats_ki.append({})
            feats_k.append(feats_ki)
        return feats_k

    def _real_data_dict_array(self, idxs, k, k_i=0, k_p=0):
        """

        Parameters
        ----------
        idxs: list or np.ndarray
            the indices of the neighs in orther to get the associated features.
            The standards are: (ks, iss_i, nei)
        k: int
            the global perturbation index.
        k_i: int (default=0)
            the perturbation index of a given perturbation type.
        k_p: int (default=0)
            the perturbation index type index.

        Returns
        -------
        feats_k: list or np.ndarray
            the descriptors of each given `k`. Standard: [iss_i][nei]{features}

        """
        feats_k = self._real_data_dict_list(idxs, k, k_i, k_p)
        return feats_k

    def _real_data_dict_list(self, idxs, k, k_i=0, k_p=0):
        """

        Parameters
        ----------
        idxs: list or np.ndarray
            the indices of the neighs in orther to get the associated features.
            The standards are: (ks, iss_i, nei)
        k: int
            the global perturbation index.
        k_i: int (default=0)
            the perturbation index of a given perturbation type.
        k_p: int (default=0)
            the perturbation index type index.

        Returns
        -------
        feats_k: list or np.ndarray
            the descriptors of each given `k`. Standard: [iss_i][nei]{features}

        """
        feats_k = []
        for i in range(len(idxs[k])):
            feats_nei = []
            for nei in range(len(idxs[k][i])):
                feats_nei.append(self.features[idxs[k][i][nei]])
            feats_k.append(feats_nei)
        return feats_k

    ###########################################################################

    ################################# Getters #################################
    ###########################################################################
    def _get_feats_k(self, k, idxs):
        """Interaction with the stored features.

        Parameters
        ----------
        k: int
            the global perturbation index.
        idxs: list or np.ndarray
            the indices of the neighs in orther to get the associated features.
            The standards are: [ks][iss_i][nei]

        Returns
        -------
        feats_k: list or np.ndarray
            the descriptors of each given `k`.

        """
        ## Applying k map for perturbations
        k_p, k_i = self._map_perturb(k)
        ## Not perturbed k
        if k_p == 0:
            feats_k = self._get_real_data(idxs, k)
        ## Virtual perturbed data
        else:
            feats_k = self._get_virtual_data(idxs, k, k_i, k_p)
        return feats_k

    ################################ Formatters ###############################
    ###########################################################################
    def _format_features(self, features):
        """Format features.

        Parameters
        ----------
        features: list or np.ndarray
            the features stored. [iss][feats]. They have to have
            deep=2.[iss][feats].

        """
        if type(features) == np.ndarray:
            sh = features.shape
            if len(sh) != 2:
                features = features.reshape((sh[0], 1))
            self.features = features
        else:
            assert(type(features) == list)
            assert(type(features[0]) == dict)
            self.features = features

    def _format_variables(self, names, out_features=[]):
        """Format variables.

        Parameters
        ----------
        names: list
            the list of names of the stored features.
        out_features: list (default=[])
            the output features names.

        """
        if names:
            if type(self.features) == np.ndarray:
                assert(len(names) == len(self.features[0]))
                self.variables = names
        else:
            if type(self.features) == np.ndarray:
                self.variables = list(range(len(self.features[0])))
            elif type(self.features) == list:
                names = []
                for i in range(len(self.features)):
                    names += self.features[i].keys()
                self.variables = list(set(names))
#            feats = self[([0], [0]), 0]
#            print feats
        ## Set out featurenames
        self._format_out_features(out_features)

    ######################### Perturbation management #########################
    ###########################################################################
    def _format_perturbation(self, perturbations):
        """Format initial perturbations.

        Parameters
        ----------
        perturbations: pst.BasePerturbation or list (default=None)
            the perturbations applied to that features.

        """
        if perturbations is None:
            def _map_perturb(x):
                """Map to obtain the indices of of particular perturbations.

                Parameters
                ----------
                k: int
                    the global index of perturbation.

                Returns
                -------
                map_k: tuple (k_i, k_p)
                    the map to obtain the perturbation object index and the
                    perturbation subindex.

            """
                if x != 0:
                    raise IndexError("Not perturbation available.")
                return 0, 0
            self._map_perturb = _map_perturb
            self._dim_perturb = [1]
        else:
            self.add_perturbations(perturbations)

    def add_perturbations(self, perturbations):
        """Add perturbations.

        Parameters
        ----------
        perturbations: pst.BasePerturbation or list
            the perturbations applied to that features.

        """
        perturbations = feat_filter_perturbations(perturbations)
        if type(perturbations) == list:
            for p in perturbations:
                self._dim_perturb.append(p.k_perturb)
                self._perturbators.append(p)
                self._create_map_perturbation()

    def _create_map_perturbation(self):
        """Create the map for getting the perturbation object."""
        ## 0. Creation of the mapper array
        limits = np.cumsum([0] + list(self._dim_perturb))
        sl = [slice(limits[i], limits[i+1]) for i in range(len(limits)-1)]
        ## Build a mapper
        mapper = np.zeros((np.sum(self._dim_perturb), 2)).astype(int)
        for i in range(len(sl)):
            if self._perturbators[i]._perturbtype != 'none':
                inds = np.zeros((sl[i].stop-sl[i].start, 2))
                inds[:, 0] = i
                inds[:, 1] = np.arange(sl[i].stop-sl[i].start)
                mapper[sl[i]] = inds

        ## 1. Creation of the mapper function
        def map_perturb(x):
            """Map to obtain the indices of of particular perturbations.

            Parameters
            ----------
            k: int
                the global index of perturbation.

            Returns
            -------
            map_k: tuple (k_i, k_p)
                the map to obtain the perturbation object index and the
                perturbation subindex.

            """
            if x < 0:
                raise IndexError("Negative numbers can not be indices.")
            if x > self.k_perturb:
                msg = "Out of bounds. There are only %s perturbations."
                raise IndexError(msg % str(self.k_perturb))
            return mapper[x]
        ## 2. Storing mapper function
        self._map_perturb = map_perturb


class ExplicitFeatures(BaseFeatures):
    """Explicit features class. In this class we have explicit representation
    of the features.

    The standarts over this data are:
    * [ks][iss]{feats}
    * (iss, feats, ks)

    """
    "TODO: adaptation of not only np.ndarray format"
    ## Type
    typefeat = 'explicit'

    def _initialization(self):
#        self._descriptormodel = lambda x, d: x
        self.possible_regions = None
        self.indices = []
        ## Default mutable functions
        self._get_real_data = self._real_data_general
        self._get_real_data = self._real_data_general
        self._get_virtual_data = self._virtual_data_general

    def __init__(self, aggfeatures, descriptormodel=None, out_type='ndarray',
                 names=[], out_features=[], nullvalue=None, indices=None):
        """The precomputed perturbations of each elements.

        Parameters
        ----------
        aggfeatures: list or np.ndarray
            the features stored. [ks][iss][feats] or (iss, feats, ks)
        descriptormodel: pst.BaseDescriptormodel (default=None)
            the descriptormodel information.
        out_type: str, optional (default='ndarray')
            the output type we want to retrieve descriptors.
        names: list (default=[])
            the stored features names.
        out_features: list (default=[])
            the output features names.
        nullvalue: float (default=None)
            the null value of the elements which not appear.
        indices: np.ndarray or list or None (default=None)
            the reindices of each index iss.

        """
        self._global_initialization()
        self._initialization()
        self._format_outtypes(out_type)
        self._format_aggfeatures(aggfeatures, indices)
        self._format_descriptormodel(descriptormodel, out_features, out_type)
        self._format_variables(names, out_features)
        self._nullvalue = self._nullvalue if nullvalue is None else nullvalue

    @property
    def shape(self):
        """Size information of the features.

        Returns
        -------
        n: int
            the number of instances of the stored features.
        nfeats: int
            the number of features variables stored.
        ks: int
            the number of perturbations plus the unperturbated.

        """
        if type(self.features) == list:
            return len(self.features[0]), len(self.variables), self.k_perturb+1
        else:
            return (len(self.features), len(self.variables), self.k_perturb+1)

    def export_features(self):
        """Export features to be copied or reinstantiated.

        Returns
        -------
        object_feats: class
            the class of the features to be instantiated.
        core_features: np.ndarray or list
            the features stored.
        pars_fea_o_in: dict
            the parameters needed to reinstantiate the same object.

        """
        object_feats = ExplicitFeatures
        core_features = self.features
        pars_fea_o_in = {}
        pars_fea_o_in['nullvalue'] = self._nullvalue
        pars_fea_o_in['names'] = self.variables
        pars_fea_o_in['descriptormodel'] = self.descriptormodel
        return object_feats, core_features, pars_fea_o_in

    ############################### Interaction ###############################
    ###########################################################################
    ################################ Candidates ###############################
    def _real_data_array_array(self, idxs, k, k_i=0, k_p=0):
        """Real data array.

        Parameters
        ----------
        idxs: list or np.ndarray
            the indices of the neighs in orther to get the associated features.
            The standards are: (ks, iss_i, nei)
        k: int
            the global perturbation index.
        k_i: int (default=0)
            the perturbation index of a given perturbation type.
        k_p: int (default=0)
            the perturbation index type index.

        Returns
        -------
        feats_k: list or np.ndarray
            the descriptors of each given `k`. Standard: (iss_i, nei, features)

        """
        feats_k = self.features[:, :, k][idxs[k]]
        assert(len(feats_k.shape) == 3)
        return feats_k

    def _real_data_array_list(self, idxs, k, k_i=0, k_p=0):
        """

        Parameters
        ----------
        idxs: list or np.ndarray
            the indices of the neighs in orther to get the associated features.
            The standards are: [ks][iss_i][nei]
        k: int
            the global perturbation index.
        k_i: int (default=0)
            the perturbation index of a given perturbation type.
        k_p: int (default=0)
            the perturbation index type index.

        Returns
        -------
        feats_k: list or np.ndarray
            the descriptors of each given `k`. Standard: [iss_i](nei, features)

        """
        feats_k = []
        for i in range(len(idxs[k])):
            feats_k.append(self.features[:, :, k][idxs[k][i]])
#            feats_ki = []
#            for nei in range(len(idxs[k][i])):
#                feats_k.append(self.features[:, :, k][idxs[k][i]])
#            feats_k.append(feats_ki)
        return feats_k

    def _real_data_dict_array(self, idxs, k, k_i=0, k_p=0):
        """

        Parameters
        ----------
        idxs: list or np.ndarray
            the indices of the neighs in orther to get the associated features.
            The standards are: (ks, iss_i, nei)
        k: int
            the global perturbation index.
        k_i: int (default=0)
            the perturbation index of a given perturbation type.
        k_p: int (default=0)
            the perturbation index type index.

        Returns
        -------
        feats_k: list or np.ndarray
            the descriptors of each given `k`. Standard: [iss_i][nei]{features}

        """
        feats_k = []
        for i in range(len(idxs[k])):
            feats_ki = []
            for nei in range(len(idxs[k][i])):
                feats_ki.append(self.features[k][idxs[k, i, nei]])
            feats_k.append(feats_ki)
        return feats_k

    def _real_data_dict_list(self, idxs, k, k_i=0, k_p=0):
        """

        Parameters
        ----------
        idxs: list or np.ndarray
            the indices of the neighs in orther to get the associated features.
            The standards are: [ks][iss_i][nei]
        k: int
            the global perturbation index.
        k_i: int (default=0)
            the perturbation index of a given perturbation type.
        k_p: int (default=0)
            the perturbation index type index.

        Returns
        -------
        feats_k: list or np.ndarray
            the descriptors of each given `k`. Standard: [iss_i][nei]{features}

        """
        feats_k = []
        for i in range(len(idxs[k])):
            feats_ki = []
            for nei in range(len(idxs[k][i])):
                feats_ki.append(self.features[k][idxs[k][i][nei]])
            feats_k.append(feats_ki)
        return feats_k

#    def _virtual_data_dict_array(self, idxs, k, k_i, k_p):
#        """
#        * idxs: (ks, iss_i, nei)
#        * feats_k: [iss_i][nei]{features}
#        """
#        feats_k = self._virtual_data_dict_list(idxs, k, k_i, k_p)
#        return feats_k
#
#    def _virtual_data_dict_list(self, idxs, k, k_i, k_p):
#        """
#        * idxs: (ks, iss_i, nei)
#        * feats_k: [iss_i][nei]{features}
#        """
#        feats_k = []
#        print idxs, k, k_i, k_p
#        for i in range(len(idxs[k_i])):
#            # Perturbation indices
#            new_idxs = self._perturbators[k_p].apply2indice(idxs[k_i][i], k_i)
#            # Indices in bounds
#            yes_idxs = [j for j in range(len(new_idxs))
#                        if new_idxs[j] < len(self.features[0])]
#            # Features
#            feats_ki = []
#            for j in range(len(new_idxs)):
#                if j in yes_idxs:
#                    aux_feat = self._perturbators[k_p].\
#                        apply2features_ind(self.features[k_p], new_idxs[j],
#                        k_i)
#                    feats_ki.append(aux_feat)
#                else:
#                    feats_ki.append({})
#            feats_k.append(feats_ki)
#        return feats_k

    ###########################################################################

    ################################# Getters #################################
    ###########################################################################
    def _get_feats_k(self, k, idxs):
        """Interaction with the stored features.

        Parameters
        ----------
        k: int
            the global perturbation index.
        idxs: list or np.ndarray
            the indices of the neighs in orther to get the associated features.
            The standards are: [ks][iss_i][nei]

        Returns
        -------
        feats_k: list or np.ndarray
            the descriptors of each given `k`.

        """
        ## Not perturbed k
        ## TODO: Extension for perturbated
        feats_k = self._get_real_data(idxs, k)
        return feats_k

    ################################ Formatters ###############################
    ###########################################################################
    def _format_aggfeatures(self, aggfeatures, indices):
        """Formatter for aggfeatures.

        Parameters
        ----------
        aggfeatures: list or np.ndarray
            the features stored. [ks][iss][feats] or (iss, feats, ks)
        indices: np.ndarray or list or None (default=None)
            the reindices of each index iss.

        """
        if type(aggfeatures) == list:
            self._k_reindices = len(aggfeatures)
            self.features = aggfeatures
            self.k_perturb = self._k_reindices-1
            # Default listdict descriptormodel
#            self._descriptormodel = lambda x, d: [e[0] for e in x]
        elif type(aggfeatures) == np.ndarray:
            if len(aggfeatures.shape) == 1:
                self._k_reindices = 1
                aggfeatures = aggfeatures.reshape((len(aggfeatures), 1, 1))
                self.features = aggfeatures
                self.k_perturb = aggfeatures.shape[2]-1
            elif len(aggfeatures.shape) == 2:
                self._k_reindices = 1
                aggfeatures = aggfeatures.reshape((len(aggfeatures),
                                                  aggfeatures.shape[1], 1))
                self.features = aggfeatures
                self.k_perturb = aggfeatures.shape[2]-1
            elif len(aggfeatures.shape) == 3:
                self._k_reindices = aggfeatures.shape[2]
                self.features = aggfeatures
                self.k_perturb = aggfeatures.shape[2]-1
            elif len(aggfeatures.shape) > 3:
                raise IndexError("Aggfeatures with more than 3 dimensions.")
        self.indices = indices

    def _format_variables(self, names, out_features=[]):
        """Format variables.

        Parameters
        ----------
        names: list (default=[])
            the stored features names.
        out_features: list (default=[])
            the output features names.

        """
        if names:
            if type(self.features) == np.ndarray:
                assert(len(names) == self.features.shape[1])
                self.variables = names
        else:
            if type(self.features) == np.ndarray:
                self.variables = list(range(self.features.shape[1]))
            elif type(self.features) == list:
                names = []
                for i in range(len(self.features[0])):
                    names += self.features[0][i].keys()
                self.variables = list(set(names))
#            feats = self[([0], [0]), 0]
#            print feats
        ## Set out featurenames
        self._format_out_features(out_features)

    ######################### Perturbation management #########################
    ###########################################################################
    def add_perturbations(self, perturbations):
        """Add perturbations.

        Parameters
        ----------
        perturbations: pst.BasePerturbation or list
            the perturbations applied to that features.

        """
#        msg = "Aggregated features can not be perturbated."
#        msg += "Change order of aggregation."
#        raise Exception(msg)
        raise NotImplementedError("Not implemented yet.")


class PhantomFeatures(BaseFeatures):
    """Phantom features class. In this 'empty' class we don't have features
    stored but a way to transform neighs_info into features using
    descriptormodels.
    """
    ## Type
    typefeat = 'phantom'

    def _initialization(self):
        ## Specific class parameters
        self.relabel_indices = None  # TODO
        self.features_array = None
        self.features = 0, 0

    def __init__(self, features_info, descriptormodel=None, perturbations=None,
                 out_type='ndarray', names=[], out_features=[]):
        """The phantom features.

        Parameters
        ----------
        features_info: tuple
            size information of the stored features.
        descriptormodel: pst.BaseDescriptormodel (default=None)
            the descriptormodel information.
        perturbations: pst.BasePerturbation (default=None)
            the perturbations applied to that features.
        out_type: str, optional (default='ndarray')
            the output type we want to retrieve descriptors.
        names: list (default=[])
            the stored features names.
        out_features: list (default=[])
            the output features names.

        """
        self._global_initialization()
        self._initialization()
        self._format_outtypes(out_type)
        self._format_features(features_info)
        self._format_descriptormodel(descriptormodel, out_features, out_type)
        self._format_variables(names, out_features)
        self._format_perturbation(perturbations)

    @property
    def k_perturb(self):
        """Number of perturbations."""
        if self._dim_perturb:
            return np.sum(self._dim_perturb)-1

    @property
    def shape(self):
        """Size information of the features.

        Returns
        -------
        n: int
            the number of instances of the stored features.
        nfeats: int
            the number of features variables stored.
        ks: int
            the number of perturbations plus the unperturbated.

        """
        n = self._n
        nfeats = self._nfeats
        ks = self.k_perturb+1
        return n, nfeats, ks

    def export_features(self):
        """Export features to be copied or reinstantiated.

        Returns
        -------
        object_feats: class
            the class of the features to be instantiated.
        core_features: np.ndarray or list
            the features stored.
        pars_fea_o_in: dict
            the parameters needed to reinstantiate the same object.

        """
        object_feats = PhantomFeatures
        core_features = self.features
        pars_fea_o_in = {}
        pars_fea_o_in['perturbations'] = self._perturbators
        pars_fea_o_in['names'] = self.variables
        pars_fea_o_in['descriptormodel'] = self.descriptormodel
        return object_feats, core_features, pars_fea_o_in

    ############################### Interaction ###############################
    ###########################################################################
    ################################# Getters #################################
    def _get_feats_k(self, k, idxs):
        """Interaction with the stored features.

        Parameters
        ----------
        k: int
            the global perturbation index.
        idxs: list or np.ndarray
            the indices of the neighs in orther to get the associated features.
            The standards are: [ks][iss_i][nei]

        Returns
        -------
        feats_k: list or np.ndarray
            the descriptors of each given `k`.

        """
        return idxs[k]

    def _get_relpos_k(self, k, d):
        """Get the relative position of the `k` perturbation.

        Parameters
        ----------
        k: int
            the global perturbation index.
        d: list or np.ndarray
            the spatial relative positions of the neighbours.

        Returns
        -------
        d_k: list or np.ndarray
            the spatial relative positions of the neighbours in the
            perturbation `k`.

        """
        return d[k]

    ################################ Formatters ###############################
    ###########################################################################
    def _format_features(self, features_info):
        """Format features.

        Parameters
        ----------
        features_info: tuple
            the features information.

        """
        self._n, self._nfeats = None, None
        if type(features_info) == tuple:
            self._n, self._vars = features_info
            self.features = features_info

    def _format_perturbation(self, perturbations):
        """Format perturbations.

        Parameters
        ----------
        perturbations: pst.BasePerturbation (default=None)
            the perturbations applied to that features.

        """
        if type(perturbations) != list:
            perturbations = [perturbations]
        self._perturbators = perturbations

    def _format_variables(self, names, out_features=[]):
        """Format variables.

        Parameters
        ----------
        names: list (default=[])
            the stored features names.
        out_features: list (default=[])
            the output features names.

        """
        if names:
            self.variables = names
        else:
            self.variables = range(self._vars)
        ## Outfeatures preparation
        self._format_out_features(out_features)
        self._nfeats = len(self.out_features)

    def _format_descriptormodel(self, descriptormodel, out_features, out_type):
        """Format descriptormodel.

        Parameters
        ----------
        descriptormodel: pst.BaseDescriptormodel (default=None)
            the descriptormodel information.
        out_features: list (default=[])
            the output features names.
        out_type: str, optional (default='ndarray')
            the output type we want to retrieve descriptors.

        """
        if descriptormodel is not None:
            self.set_descriptormodel(descriptormodel)
        else:
            # Default descriptormodel
            self.set_descriptormodel(DistancesDescriptor(self._vars))


###############################################################################
######################### Auxiliar Features functions #########################
###############################################################################
def checker_sp_descriptor(retriever, features_o):
    pass


def _featuresobject_parsing_creation(feats_info):
    """Feature object information parsing and instantiation.

    Parameters
    ----------
    feats_info: pst.BaseFeatures, np.ndarray, tuple
        the features information to insntantiate a features object. The
        standard inputs are:
            * np.ndarray
            * Features object
            * (Features object, descriptormodel)
            * (np.ndarray, descriptormodel)

    Returns
    -------
    feats_info: pst.BaseFeatures
        the features insntance.

    TODO
    ----
    Give support to listFeatures.
    """
    if type(feats_info) == np.ndarray:
        sh = feats_info.shape
        if len(sh) == 1:
            feats_info = feats_info.reshape((sh[0], 1))
            feats_info = ImplicitFeatures(feats_info)
        if len(sh) == 2:
            feats_info = ImplicitFeatures(feats_info)
        elif len(sh) == 3:
            feats_info = ExplicitFeatures(feats_info)
    elif isinstance(feats_info, BaseFeatures):
        pass
    else:
        assert(type(feats_info) == tuple)
        assert(type(feats_info[0]) == np.ndarray)
        assert(type(feats_info[1]) == dict)
        if len(feats_info) == 2:
            pars_feats = feats_info[1]
        else:
            assert(len(feats_info) == 3)
            pars_feats = feats_info[1]
            pars_feats['descriptormodel'] = feats_info[2]
        sh = feats_info[0].shape
        if len(sh) == 1:
            feats_arr = feats_info[0].reshape((sh[0], 1))
            feats_info = ImplicitFeatures(feats_arr, **pars_feats)
        if len(sh) == 2:
            feats_info = ImplicitFeatures(feats_info[0], **pars_feats)
        elif len(sh) == 3:
            feats_info = ExplicitFeatures(feats_info[0], **pars_feats)
    assert(isinstance(feats_info, BaseFeatures))
    return feats_info
