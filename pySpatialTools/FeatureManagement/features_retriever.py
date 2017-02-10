
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
from pySpatialTools.utils.mapper_vals_i import create_mapper_vals_i
from pySpatialTools.utils.selectors import Feat_RetrieverSelector,\
    format_selection
from pySpatialTools.utils.neighs_info import ensuring_neighs_info
#from aux_featuremanagement import create_aggfeatures
from features_objects import _featuresobject_parsing_creation, BaseFeatures
from Descriptors import DummyDescriptor, BaseDescriptorModel
from aux_resulter_building import DefaultResulter


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
#        self._variables = {}       ## TO CHANGE
#        self.featuresnames = []    ## TO CHANGE
        self.out_features = []
        self.k_perturb = 0
        self._out = 'ndarray'  # dict
        ## IO managers
        self._maps_input = []
        self._maps_output = None
        self._maps_vals_i = None

    def __init__(self, features_objects, mode=None, maps_input=None,
                 maps_output=None, maps_vals_i=None, descriptormodels=None,
                 selectors=[None]*3, resulter=None):
        """Manager of features.

        Parameters
        ----------
        features_objects: list or pst.BaseFeatures
            the features objects in order to be managed by the manage.
        mode: str optional or None (default=None)
            the mode we want to manage the different features. 'parallel' for
            a parallel management or 'sequential' for a sequential management.
        maps_input: function or pst.BaseSelector (default=None)
            input map for the indices of transformed elements.
        maps_output: function or pst.BaseSelector (default=None)
            output map for the descriptors computed by the descriptormodels.
        maps_vals_i: function or pst.BaseSelector (default=None)
            the mapper from elements `i` to the values of the storing the
            resulter measure `vals_i`.
        descriptormodels: list or pst.BaseDescriptormodel (default=None)
            the descriptormodels to be manage externally by that class.
        selectors: list, tuple or pst.BaseSelector (default=[None]*3)
            the selection information to manage the process.
        resulter: pst.BaseResulter (default=None)
            the object which manages all the possibilities of measure
            construction we could do.

        """
        self._initialization()
#        out = out if out in ['ndarray', 'dict'] else None
#        self._out = self._out if out is None else out
        self._format_descriptormodel(descriptormodels)
        self._format_maps(maps_input, maps_output, maps_vals_i)
        self._format_features(features_objects)
        self._format_mode(mode)
        self._format_outfeatures()
        self.set_selector(*selectors)
        self._format_result_building(resulter)

    def __getitem__(self, i_feat):
        if i_feat < 0 or i_feat >= len(self.features):
            raise IndexError("Not correct index for features.")
        return self.features[i_feat]

    def __len__(self):
        return len(self.features)

    @property
    def shape(self):
        """As a mapper its shapes represents the size of the input and the
        output stored in maps_vals_i.

        Returns
        -------
        n_in: int or None
            the number of values indices of the elements to retrieve and
            compute their descriptors. If it is None, it is open (e.g. on-line
            elements added)
        n_out: int or None
            the number of values indices of the resulter measure. In case of
            None, it is an open container.

        """
        return self._maps_vals_i.n_in, self._maps_vals_i.n_out

    @property
    def shape_measure(self):
        """The measures of the possible output measure.

        Returns
        -------
        n_vals_i: int or None
            the number of values indices of the resulter measure. In case of
            None, it is an open container.
        n_feats: int or None
            the number of descriptors resultant in the measure.
        ks: int
            the number of perturbations plus the non-perturbated case.

        """
        n_vals_i = self._maps_vals_i.n_out
        n_feats = len(self.out_features) if self.out_features else None
        return n_vals_i, n_feats, self.k_perturb+1

#    @property
#    def nfeats(self):
#        return len(self.variables)

    ################################ Formatters ###############################
    ###########################################################################
    ############################## Format features ############################
    def _format_features(self, features_objects):
        """Formatter of features.

        Parameters
        ----------
        features_objects: list or pst.BaseFeatures
            the features objects in order to be managed by the manage.

        """
        ## 0. Format to list mode
        # Format to list mode
        if type(features_objects) != list:
            features_objects = [features_objects]
        # Format to feature objects
        nfeat = len(features_objects)
        for i in range(nfeat):
#            features_objects[i] = self._auxformat_features(features_objects[i])
            features_objects[i] =\
                _featuresobject_parsing_creation(features_objects[i])
        self.features = features_objects
        ## 1. Check input
        if nfeat == 0:
            msg = "Any feature object is input in the featureRetriever."
            raise TypeError(msg)
        ## 2. Format kperturb
        self._format_k_perturbs()

    def _format_descriptormodel(self, descriptormodels=None):
        """Formatter of the descriptormodels.

        Parameters
        ----------
        descriptormodels: pst.BaseDescriptormodel (default=None)
            the descriptormodels to manage externally of the features classes.

        """
        if descriptormodels is None:
            self.descriptormodels = [DummyDescriptor()]
        else:
            if type(descriptormodels) != list:
                descriptormodels = [descriptormodels]
            self.descriptormodels += descriptormodels

    def _format_result_building(self, resulter=None):
        """Format and setting the resulter. The resulter manages all the
        possibilities of measure construction we could do.

        Parameters
        ----------
        resulter: pst.BaseResulter
            the object which manages all the possibilities of measure
            construction we could do.

        """
        self.resulter = DefaultResulter(self, resulter)

    def _format_k_perturbs(self):
        """Format k perturbations."""
        ## 1. Format kperturb
        kp = self[0].k_perturb
        k_rei_bool = [self[i].k_perturb == kp for i in range(len(self))]
        # Check k perturbations
        if not all(k_rei_bool):
            msg = "Not all the feature objects have the same perturbations."
            raise Exception(msg)
        self.k_perturb = kp

    ############################## Format IO maps #############################
    def _format_map_vals_i(self, sp_typemodel):
        """Format mapper to indicate external val_i to aggregate result.

        Parameters
        ----------
        sp_typemodel: list, tuple, np.ndarray
            the information to set the map_vals_i in order to obtain the
            stored index for each element.

        """
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
        """Formatter of maps.

        Parameters
        ----------
        maps_input: function or pst.BaseSelector
            input map for the indices of transformed elements.
        maps_output: function or pst.BaseSelector
            output map for the descriptors computed by the descriptormodels.
        maps_vals_i: function or pst.BaseSelector
            the mapper from elements `i` to the values of the storing the
            resulter measure `vals_i`.

        """
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

        Parameters
        ----------
        mode: str optional or None (default=None)
            the mode we want to manage the different features. 'parallel' for
            a parallel management or 'sequential' for a sequential management.

        """
        ## Format basic modes
        if mode is None:
            ## Out setting
            assert(all([self[0]._out == e._out for e in self]))
            self._out = self[0]._out
            ## If open out_features
            if not len(self[0].out_features):
                n_f = len(self)
                logi = [not len(self[i].out_features) for i in range(n_f)]
                assert(all(logi))
                self.out_features = []
                self.mode = None
            else:
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
            ## TODO: parallel and sequential TODO
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
        """Programable get_type_feats. It sets selection functions and
        parameters.

        Parameters
        ----------
        selector1: tuple, np.ndarray or None (default=None)
            the selection of the features for the element `i`.
        selector2: tuple, np.ndarray or None (default=None)
            the selection of the features for the neighs of the element `i`.
        selector3: tuple, np.ndarray or None (default=None)
            the selection of the descriptor models managed by that class.

        """
        if selector1 is None:
            self.get_type_feats = self._general_get_type_feat
            self._get_input_features = self._get_input_features_general
            self._get_output_features = self._get_output_features_general
            self._complete_desc_i = self._complete_desc_i_general
        else:
            typ = type(selector1)
            if selector2 is not None:
                assert((type(selector2) == typ) and (type(selector3) == typ))
            if typ == tuple:
                if selector2 is None:
                    selector1, selector2, selector3 = selector1
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
        """Set how it maps each element of the input indices has to be
        transformed in order to be obtain the stored index for each element.

        Parameters
        ----------
        maps_vals_i: function or pst.BaseSelector
            the mapper from elements `i` to the values of the storing the
            resulter measure `vals_i`.

        """
        #self._maps_vals_i = _maps_vals_i
        if type(_maps_vals_i) in [int, slice]:
            _maps_vals_i = (self._maps_vals_i, _maps_vals_i)
            self._format_map_vals_i(_maps_vals_i)
        self._format_map_vals_i(_maps_vals_i)

    def set_descriptormodels(self, descriptormodels):
        """Set descriptormodels.

        Parameters
        ----------
        descriptormodels: list or pst.BaseDescriptormodel
            the descriptormodels to be manage externally by that class.

        """
        self._format_descriptormodel(descriptormodels)

    def set_selector(self, selector1, selector2=None, selector3=None):
        """Programable get_type_feats. It sets selection functions and
        parameters.

        Parameters
        ----------
        selector1: tuple, np.ndarray or None (default=None)
            the selection of the features for the element `i`.
        selector2: tuple, np.ndarray or None (default=None)
            the selection of the features for the neighs of the element `i`.
        selector3: tuple, np.ndarray or None (default=None)
            the selection of the descriptor models managed by that class.

        """
        self._format_selector(selector1, selector2, selector3)

    ################################# Getters #################################
    ###########################################################################
    def compute_descriptors(self, i, neighs_info, k=None, feat_selectors=None):
        """General compute descriptors for descriptormodel class.

        Parameters
        ----------
        i: np.ndarray or list
            the indices of the elements `i`.
        neighs_info: pst.Neighs_Info
            the container of all the information of the neighbourhood.
        k: int or list (default=None)
            the perturbations indices we wantto get.
        feat_selectors: list, tuple or pst.BaseSelector (default=None)
            the selection information.

        Returns
        -------
        descriptors: list
            the descriptors for each perturbation and element.
        vals_i: list or np.ndarray
            the store information index of each element `i`.

        """
        ## 0. Prepare list of k
        ks = list(range(self.k_perturb+1)) if k is None else k
        ks = [ks] if type(ks) == int else ks
        i_input = [i] if type(i) == int else i
#        if type(feat_selectors) == tuple:
#            feat_selectors = [feat_selectors]
        neighs_info = ensuring_neighs_info(neighs_info, k)
#        sh = neighs_info.shape
#        print sh, len(i_input), len(ks), ks, i_input
#        assert(len(i_input) == sh[1])
#        assert(len(ks) == sh[0])
        ## 1. Prepare selectors
        t_feat_in, t_feat_out, t_feat_des =\
            self.get_type_feats(i, feat_selectors)
#        print t_feat_in, t_feat_out, t_feat_des
        ## 2. Get pfeats (pfeats 2dim array (krein, jvars))
        desc_i = self._get_input_features(i_input, ks, t_feat_in)
#        print i_input, ks, self._get_input_features, type(t_feat_in)
#        print '.'*20, desc_i
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

    ########################## Main manager functions #########################
    ## Interaction with resulter object
    def initialization_desc(self):
        """Wrapper to the resulter initialization descriptor measure function.

        Returns
        -------
        null_descriptor: dict, list or np.ndarray
            the empty null descriptor for the given measure.

        """
        return self.resulter.initialization_desc()

    def initialization_output(self):
        """Wrapper to the resulter initialization measure resultant.

        Returns
        -------
        measure: np.ndarray or list
            the transformed measure computed by the whole spatial descriptor
            model.

        """
        return self.resulter.initialization_output()

    def _join_descriptors(self, descriptors):
        """Wrapper to the resulter initialization measure resultant.

        Parameters
        ----------
        descriptors: list
            the final descriptors after computing the associated union of the
            descriptors of `i` with the ones of its neighbourhood.

        Returns
        -------
        descriptors: list
            the final descriptors after formatting properly.

        """
        return self.resulter._join_descriptors(descriptors)

    def add2result(self, measure, desc_i, vals_i):
        """Wrapper to the resulter for adding the new descriptors computed to
        the resultant measure.

        Parameters
        ----------
        measure: np.ndarray or list
            the transformed measure computed by the whole spatial descriptor
            model.
        desc_i: np.ndarray, list or dict or others
            the spatial descriptors associated to the element `i`.
        vals_i: list or np.ndarray
            the store information index of each element `i`.

        Returns
        -------
        measure: np.ndarray or list
            the transformed measure computed by the whole spatial descriptor
            model.

        """
        return self.resulter.add2result(measure, desc_i, vals_i)

    def to_complete_measure(self, measure):
        """Wrapper to the resulter for the completing function.

        Parameters
        ----------
        measure: np.ndarray or list
            the measure computed by the whole spatial descriptor model.

        Returns
        -------
        measure: np.ndarray or list
            the transformed measure computed by the whole spatial descriptor
            model.

        """
        return self.resulter.to_complete_measure(measure)

    ######################## Interaction with features ########################
    ############################# Input features ##############################
    def _get_input_features_constant(self, i, k, typefeats=(0, 0)):
        """Get 'input' features. Get the features of the elements of which we
        want to study their neighbourhood. We consider constant typefeats.
        That case is under the assumption of constant selection of features.

        Parameters
        ----------
        i: int, np.ndarray or list
            the indices of the elements.
        k: list
            perturbations indices we want to get.
        typefeats: tuple
            the selectors for the features of the elements `i`.

        Returns
        -------
        feats_i: np.ndarray or list
            the features of the elements `i`.

        """
        ## Input mapping
        i_input = self._maps_input[typefeats[0]](i)
        ## Retrieve features
#        print '-'*20, i_input, k
        feats_i = self.features[typefeats[1]].compute((i_input, k))
#        print '-.'*20, feats_i
        ## Outformat
        feats_i = self._maps_output(self, feats_i)
#        print '-,'*20, feats_i
        return feats_i

    def _get_input_features_variable(self, i, k, typefeats=(0, 0)):
        """Get 'input' features. Get the features of the elements of which we
        want to study their neighbourhood. We consider constant typefeats.
        That case is under the assumption of variable selection of features.

        Parameters
        ----------
        i: int, np.ndarray or list
            the indices of the elements.
        k: list
            perturbations indices we want to get.
        typefeats: tuple or list
            the selectors for the features of the elements `i`.

        Returns
        -------
        feats_i: np.ndarray or list
            the features of the elements `i`.

        """
        ## Preparing input
        i = [[i]] if type(i) == int else i
        i_l = len(i)
        k_l = 1 if type(k) == int else len(k)
        typefeats = [typefeats]*i_l if type(typefeats) == tuple else typefeats
        feats_i = [[] for kl in range(k_l)]
#        print '`'*20, i_l
        for j in range(i_l):
            ## Input mapping
            i_j = [i[j]] if type(i[j]) == int else i[j]
            i_input = self._maps_input[typefeats[j][0]](i_j)
            ## Retrieve features
            feats_ij = self.features[typefeats[j][1]].compute((i_input, k))
#            print feats_ij, i_input, k, typefeats, j
            ## Outformat
            feats_ij = self._maps_output(self, feats_ij)
#            print feats_ij
            for k_j in range(k_l):
                feats_i[k_j].append(feats_ij[k_j][0])
        assert(len(feats_i) == k_l)
        assert(all([len(feats_i[ind]) == i_l for ind in range(k_l)]))
        return feats_i

    def _get_input_features_general(self, i, k, typefeats=(0, 0)):
        """Get 'input' features. Get the features of the elements of which we
        want to study their neighbourhood. We dont consider anything.
        That case is under no assumtions of selection.

        Parameters
        ----------
        i: int, np.ndarray or list
            the indices of the elements.
        k: list or np.ndarray
            perturbations indices we want to get.
        typefeats: tuple or list
            the selectors for the features of the elements `i`.

        Returns
        -------
        feats_i: np.ndarray or list
            the features of the elements `i`.

        """
        if type(typefeats) == list:
            feats_i = self._get_input_features_variable(i, k, typefeats)
        else:
            feats_i = self._get_input_features_constant(i, k, typefeats)
        return feats_i

    ############################# Output features #############################
    def _get_output_features_constant(self, neighs_info, k, typefeats=(0, 0)):
        """Get 'output' features. Get the features of the elements in the
        neighbourhood of the elements we want to study. That case is under
        assumption of the constant selection of features.

        Parameters
        ----------
        neighs_info: pst.Neighs_Info
            the container of all the information of the neighbourhood.
        k: list or np.ndarray
            perturbations indices we want to get.
        typefeats: tuple or list
            the selectors for the features of the neighs of elements `i`.

        Returns
        -------
        feats_neighs: np.ndarray or list
            the features of the neighs of the elements `i`.

        """
        ## Neighs info as an object
        neighs_info = ensuring_neighs_info(neighs_info, k)
        ## Input mapping
        neighs_info = self._maps_input[typefeats[0]](neighs_info)
        ## Features retrieve
        feats_neighs = self.features[typefeats[1]].compute((neighs_info, k))
        ## Outformat
        feats_neighs = self._maps_output(self, feats_neighs)
        return feats_neighs

    def _get_output_features_variable(self, neighs_info, k, typefeats=(0, 0)):
        """Get 'output' features. Get the features of the elements in the
        neighbourhood of the elements we want to study. That case is under
        assumption of the variable selection of features.

        Parameters
        ----------
        neighs_info: pst.Neighs_Info
            the container of all the information of the neighbourhood.
        k: list or np.ndarray
            perturbations indices we want to get.
        typefeats: tuple or list
            the selectors for the features of the neighs of elements `i`.

        Returns
        -------
        feats_neighs: np.ndarray or list
            the features of the neighs of the elements `i`.

        """
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
            feats_neighs_j =\
                self.features[typefeats[j][1]].compute((neighs_info_j, k))
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
        neighbourhood of the elements we want to study. That case is under
        no assumption of selection.

        Parameters
        ----------
        neighs_info: pst.Neighs_Info
            the container of all the information of the neighbourhood.
        k: list or np.ndarray
            perturbations indices we want to get.
        typefeats: tuple or list
            the selectors for the features of the neighs of elements `i`.

        Returns
        -------
        feats_neighs: np.ndarray or list
            the features of the neighs of the elements `i`.

        """
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
        neighbourhood features. That unction is under assumption of constant
        selection.

        Parameters
        ----------
        i: int, list or np.ndarray
            the index information.
        neighs_info:
            the neighbourhood information of each `i`.
        desc_i: np.ndarray or list of dict
            the descriptors associated to the elements `iss` for each
            perturbation `k`.
        desc_neighs: np.ndarray or list of dict
            the descriptors associated to the neighbourhood elements of each
            `iss` for each perturbation `k`.
        vals_i: list or np.ndarray
            the storable index information for each perturbation `k`.
        t_feat_desc: tuple or list
            the selection of the features and descriptormodel to compute the
            descriptor from desc_i and desc_neighs.

        Returns
        -------
        descriptors: list
            the descriptors for each perturbation and element.

        """
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
        neighbourhood features. Assumption of variable t_feat_desc. That
        function is under assumption of variable selection.

        Parameters
        ----------
        i: int, list or np.ndarray
            the index information.
        neighs_info:
            the neighbourhood information of each `i`.
        desc_i: np.ndarray or list of dict
            the descriptors associated to the elements `iss` for each
            perturbation `k`.
        desc_neighs: np.ndarray or list of dict
            the descriptors associated to the neighbourhood elements of each
            `iss` for each perturbation `k`.
        vals_i: list or np.ndarray
            the storable index information for each perturbation `k`.
        t_feat_desc: tuple or list
            the selection of the features and descriptormodel to compute the
            descriptor from desc_i and desc_neighs.

        Returns
        -------
        descriptors: list
            the descriptors for each perturbation and element.

        """
        ## Preparing input
        i_l = 1 if type(i) == int else len(i)
        i = [i]*i_l if type(i) == int else i
        k_l = len(desc_i)
        if type(t_feat_desc) != list:
            t_feat_desc = [t_feat_desc]*i_l
        ## Sequential computation
        descriptors = [[] for kl in range(k_l)]
#        print vals_i, desc_i, self.k_perturb
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
        neighbourhood features. No assumptions about iss and t_feat_desc.
        That function has not assumtions of selection features and
        descriptormodel.

        Parameters
        ----------
        i: int, list or np.ndarray
            the index information.
        neighs_info:
            the neighbourhood information of each `i`.
        desc_i: np.ndarray or list of dict
            the descriptors associated to the elements `iss` for each
            perturbation `k`.
        desc_neighs: np.ndarray or list of dict
            the descriptors associated to the neighbourhood elements of each
            `iss` for each perturbation `k`.
        vals_i: list or np.ndarray
            the storable index information for each perturbation `k`.
        t_feat_desc: tuple or list
            the selection of the features and descriptormodel to compute the
            descriptor from desc_i and desc_neighs.

        Returns
        -------
        descriptors: list
            the descriptors for each perturbation and element.

        """
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
        """Get indice to store the final result.

        Parameters
        ----------
        i: int, list or np.ndarray
            the index information.
        ks: list
            perturbations indices we want to get.

        Returns
        -------
        vals_i: list or np.ndarray
            the storable index information for each perturbation `k`.

        """
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
        """Format properly general typefeats selector information. That
        function has not assumption of selection information.

        Parameters
        ----------
        i: int, list or np.ndarray
            the index information.
        typefeats_i: list, tuple or None (default=None)
            the selector information of all the selections we have to do
            in that process for each element `i`.

        Returns
        -------
        typefeats_i: tuple or list
            the selectors for the features of the elements `i`.
        typefeats_nei: tuple or list
            the selectors for the features of the neighs of elements `i`.
        typefeats_desc: tuple or list
            the selection of the features and descriptormodel to compute the
            descriptor from desc_i and desc_neighs.

        """
        if typefeats_i is None:
            typefeats_i, typefeats_nei, typefeats_desc = self.selector
            if type(i) == list:
                typefeats_i = [typefeats_i]*len(i)
                typefeats_nei = [typefeats_nei]*len(i)
                typefeats_desc = [typefeats_desc]*len(i)
        else:
            if type(i) == list:
                typefeats_input = typefeats_i
                typefeats_i, typefeats_nei, typefeats_desc = [], [], []
                for j in range(len(i)):
                    typefeats_i.append(typefeats_input[j][0])
                    typefeats_nei.append(typefeats_input[j][1])
                    typefeats_desc.append(typefeats_input[j][2])
            else:
                typefeats_i, typefeats_nei, typefeats_desc = typefeats_i
#            if type(i) == int:
#                typefeats_i, typefeats_nei, typefeats_desc = typefeats_i
#            else:
#                typefeat = typefeats_i
#                typefeats_i, typefeats_nei, typefeats_desc = [], [], []
#                for j in range(len(i)):
#                    typefeats_i.append(typefeat[j][0])
#                    typefeats_nei.append(typefeat[j][1])
#                    typefeats_desc.append(typefeat[j][2])
        return typefeats_i, typefeats_nei, typefeats_desc

    def _static_get_type_feat(self, i, typefeats_i=None):
        """Format properly typefeats selector information. That function has
        assumptions of static selection.

        Parameters
        ----------
        i: int, list or np.ndarray
            the index information.
        typefeats_i: list, tuple or None (default=None)
            the selector information of all the selections we have to do
            in that process for each element `i`.

        Returns
        -------
        typefeats_i: tuple or list
            the selectors for the features of the elements `i`.
        typefeats_nei: tuple or list
            the selectors for the features of the neighs of elements `i`.
        typefeats_desc: tuple or list
            the selection of the features and descriptormodel to compute the
            descriptor from desc_i and desc_neighs.

        """
        typefeats_i, typefeats_nei, typefeats_desc = self.selector
#        if type(i) == list:
#            typefeats_i = [typefeats_i]*len(i)
#            typefeats_nei = [typefeats_nei]*len(i)
#            typefeats_desc = [typefeats_desc]*len(i)
        return typefeats_i, typefeats_nei, typefeats_desc

    def _selector_get_type_feat(self, i, typefeats_i=None):
        """Get information only from selector.

        Parameters
        ----------
        i: int, list or np.ndarray
            the index information.
        typefeats_i: list, tuple or None (default=None)
            the selector information of all the selections we have to do
            in that process for each element `i`.

        Returns
        -------
        typefeats_i: tuple or list
            the selectors for the features of the elements `i`.
        typefeats_nei: tuple or list
            the selectors for the features of the neighs of elements `i`.
        typefeats_desc: tuple or list
            the selection of the features and descriptormodel to compute the
            descriptor from desc_i and desc_neighs.

        """
        selector_i = format_selection(self.selector[i])
        typefeats_i, typefeats_nei, typefeats_desc = selector_i
#        if type(i) == int:
#            typefeats_i, typefeats_nei, typefeats_desc = self.selector[i]
#        else:
#            typefeats_i, typefeats_nei, typefeats_desc = [], [], []
#            selectors_i = self.selector[i]
#            print selectors_i
#            for j in range(len(i)):
#                typefeats_i.append(selectors_i[j][0])
#                typefeats_nei.append(selectors_i[j][1])
#                typefeats_desc.append(selectors_i[j][2])
        return typefeats_i, typefeats_nei, typefeats_desc

    ######################## Perturbation management ##########################
    ###########################################################################
    def add_perturbations(self, perturbations):
        """Adding perturbations to features.

        Parameters
        ----------
        perturbations: list or pst.BasePerturbation
            the perturbation information.

        """
        ## 1. Apply perturbations
        for i_ret in range(len(self.features)):
            self.features[i_ret].add_perturbations(perturbations)
        ## 2. Format kperturb
        self._format_k_perturbs()
        ## 3. Reformat resulter
        self._format_result_building()

    ######################### Aggregation management ##########################
    ###########################################################################
    def add_aggregations(self, aggfeatures):
        """Add aggregations to featuremanager. Only it is useful this function
        if there is only one retriever previously and we are aggregating the
        first one.

        Parameters
        ----------
        aggfeatures: np.ndarray or list
            the aggregated features to be stored in order to be managed.

        """
        self.features.append(aggfeatures)
#        ## 0. Get kfeats
#        kfeats = [kfeat] if type(kfeat) == int else kfeat
#        if kfeats is None:
#            kfeats = []
#            for i in range(len(self)):
#                if self.features[i].typefeat == 'implicit':
#                    kfeats.append(i)
#        ## 1. Compute and store aggregations
#        for i in kfeats:
#            aggfeatures = create_aggfeatures(discretization, regmetric,
#                                             self.features[i],
#                                             self.descriptormodel)
#            self.features.append(aggfeatures)

    ####################### Auxiliar temporal functions #######################
    ###########################################################################


###############################################################################
######################### Auxiliar Features functions #########################
###############################################################################
############################ Features parsing utils ###########################
# Utils to parse different ways to give features information and output or a
# Features instance or a FeaturesManager instance.
def _featuresmanager_parsing_creation(feats_info):
    """FeaturesManager instantiation from features information. This function
    transforms features information into a FeaturesManager object.

    Parameters
    ----------
    feats_info: np.ndarray, list, tuple, pst.BaseFeatures, pst.FeaturesManager
        The features information to create features object. The Standards
        inputs accepted are:
            * Features object
            * FeaturesManager object
            * (Features object, pars_featuresManager)
            * (Features object, pars_featuresManager, descriptormodel)

    Returns
    -------
    feats_info: pst.FeaturesManager
        the features manager created by the information given in the input.

    """
    if isinstance(feats_info, BaseFeatures):
        feats_info = FeaturesManager(feats_info)
    elif type(feats_info) == list:
        assert(all([isinstance(e, BaseFeatures) for e in feats_info]))
        feats_info = FeaturesManager(feats_info)
    elif isinstance(feats_info, FeaturesManager):
        pass
    else:
        assert(type(feats_info) == tuple)
        if isinstance(feats_info[0], BaseFeatures) or type(feats_info[0]) == list:
            if type(feats_info[0]) == list:
                assert(all([isinstance(e, BaseFeatures) for e in feats_info[0]]))
            assert(type(feats_info[1]) == dict)
            assert(len(feats_info) >= 2)
            if len(feats_info) == 2:
                pars_features = feats_info[1]
            else:
                assert(len(feats_info) == 3)
                if type(feats_info[2]) == list:
                    for i in range(len(feats_info[2])):
                        assert(isinstance(feats_info[2][i],
                                          BaseDescriptorModel))
                else:
                    assert(isinstance(feats_info[2], BaseDescriptorModel))
                pars_features = feats_info[1]
                pars_features['descriptormodels'] = feats_info[2]
            feats_info = FeaturesManager(feats_info[0], **pars_features)
        else:
            assert(type(feats_info[0]) == tuple)
            new_feats_info = _featuresobject_parsing_creation(feats_info[0])
            if len(feats_info) == 2:
                new_feats_info = (new_feats_info, feats_info[1])
            else:
                new_feats_info = (new_feats_info, feats_info[1], feats_info[2])
            feats_info = _featuresmanager_parsing_creation(new_feats_info)
    assert(isinstance(feats_info, FeaturesManager))
    return feats_info


def _features_parsing_creation(feats_info):
    """Features General parsing function. This function transforms any features
    information into a FeaturesManager object.

    Parameters
    ----------
    feats_info: np.ndarray, list, tuple, pst.BaseFeatures, pst.FeaturesManager
        The features information to create features object. The Standards
        inputs accepted are:
            * np.ndarray
            * Features object
            * FeaturesManager object
            * (np.ndarray, pars_featuresManager)
            * (np.ndarray, pars_featuresManager, descriptormodel)
            * (Features object, pars_featuresManager)
            * (Features object, pars_featuresManager, descriptormodel)
            * (Features_info tuple, pars_featuresManager)
            * (Features_info tuple, pars_featuresManager, descriptormodel)

    Returns
    -------
    feats_info: pst.FeaturesManager
        the features manager created by the information given in the input.

    """
    if type(feats_info) == np.ndarray:
        feats_info = _featuresobject_parsing_creation(feats_info)
        feats_info = FeaturesManager(feats_info)
    elif isinstance(feats_info, BaseFeatures):
        feats_info = FeaturesManager(feats_info)
    elif type(feats_info) == list:
        assert(all([isinstance(e, BaseFeatures) for e in feats_info]))
        feats_info = FeaturesManager(feats_info)
    elif isinstance(feats_info, FeaturesManager):
        pass
    else:
        assert(type(feats_info) == tuple)
        assert(type(feats_info[1]) == dict)
        if type(feats_info[0]) == np.ndarray:
            feats_info = _featuresobject_parsing_creation(feats_info)
            feats_info = FeaturesManager(feats_info)
        elif isinstance(feats_info[0], BaseFeatures):
            feats_info = _featuresmanager_parsing_creation(feats_info)
        elif type(feats_info[0]) == list:
            feats_info = _featuresmanager_parsing_creation(feats_info)
        else:
            assert(type(feats_info[0]) == tuple)
            new_feats_info = _featuresobject_parsing_creation(feats_info[0])
            if len(feats_info) == 2:
                new_pars_info = (new_feats_info, feats_info[1])
            else:
                new_pars_info = (new_feats_info, feats_info[1], feats_info[2])
            feats_info = _featuresmanager_parsing_creation(new_pars_info)
    assert(isinstance(feats_info, FeaturesManager))
    return feats_info
