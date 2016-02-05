
"""
DescriptorModel
---------------
Module which contains the abstract class method for computing descriptor
models from puntual features.

"""

import numpy as np
from pySpatialTools.Feature_engineering.features_retriever import AggFeatures,\
    FeaturesRetriever, PointFeatures
from pySpatialTools.IO import create_mapper_vals_i

from pySpatialTools.Feature_engineering.aux_descriptormodels import\
    append_addresult_function


class DescriptorModel:
    """Abstract class for desctiptor models. Contain the common functions
    that are required for each desctiptor instantiated object.
    """

    def compute_descriptors(self, i, neighs_info, k=None, typefeats=(0, 0)):
        """General compute descriptors for descriptormodel class.
        """
        ## Format properly typefeats
        typefeats = self._get_typefeats(typefeats)
        ## Get pfeats (pfeats 2dim array (krein, jvars))
        desc_i, desc_neigh = self._get_prefeatures(i, neighs_info, k,
                                                   typefeats)
        ## Map vals_i
        vals_i = self._compute_vals_i(i, k, typefeats)
        ## Complete descriptors
        descriptors = self._complete_desc_i(i, neighs_info, desc_i, desc_neigh,
                                            vals_i)
        return descriptors, vals_i

    def _get_prefeatures(self, i, neighs_info, k, typefeats):
        """General interaction with features object to get point features from
        it.
        """
        desc_i, desc_neigh =\
            self.features._get_prefeatures(i, neighs_info, k, typefeats)
        return desc_i, desc_neigh

    def _complete_desc_i(self, i, neighs_info, desc_i, desc_neighs, vals_i):
        "Dummy completion for general abstract class."
        ## TOTEST
        desc = []
        for k in xrange(len(desc_i)):
            desc.append(self.relative_descriptors(i, neighs_info, desc_i[k],
                                                  desc_neighs[k], vals_i[k]))
        descriptors = np.vstack(desc)
        return descriptors

    def relative_descriptors(self, i, neighs_info, desc_i, desc_neigh, vals_i):
        "General default relative descriptors."
        return desc_neigh

    ####################### Compulsary general functions ######################
    ###########################################################################
    def _get_typefeats(self, typefeats):
        return typefeats

    def _compute_vals_i(self, i, k, typefeats):
        return self.features._get_vals_i(i, k, typefeats)

    ################# Dummy compulsary overwritable functions #################
    def to_complete_measure(self, corr_loc):
        """Main function to compute the complete normalized measure of pjensen
        from the matrix of estimated counts.
        """
        return corr_loc

    def add2result(self, res_total, res_i):
        """Addding results to the final aggregated result. We assume here
        additivity property.
        """
        return res_total + res_i

    ###########################################################################
    ########################## Formatter functions ############################
    ###########################################################################
    def _format_features(self, features):
        "Format features."
        if type(features) == np.ndarray:
            sh = features.shape
            if len(sh) == 1:
                features = features.reshape((sh[0], 1))
                fea = self._compute_featuresnames(features)
                features = PointFeatures(features, out_features=fea,
                                         characterizer=self.compute_characs)
            if len(sh) == 2:
                fea = self._compute_featuresnames(features)
                features = PointFeatures(features, out_features=fea,
                                         characterizer=self.compute_characs)
            elif len(sh) == 3:
                features = AggFeatures(features)
            self.features = FeaturesRetriever(features)
        elif features.__name__ == "pst.FeaturesRetriever":
            self.features = features
        elif features.__name__ == 'pst.FeaturesObject':
            self.features = FeaturesRetriever(features)
        ## Setting features (linking descriptor and its featuresobjects)
        self.features.set_descriptormodel(self)

    def _compute_featuresnames(self, featureobject):
        """Compute the possible feature names from the pointfeatures."""
        if type(featureobject) == np.ndarray:
            featuresnames = self._f_default_names(featureobject)
            return featuresnames
        if '_type' in dir(featureobject):
            featuresnames = self._f_default_names(featureobject.features)
        return featuresnames

    def _format_result_building(self):
        "Format how to build and aggregate results."
        "TODO: Dict-array."
        ## Size of the possible results.
        try:
            n_vals_i = self._maps_vals_i.n_out
        except:
            n_vals_i = self.features.shape[0]
        n_feats = self.features.nfeats
        ## Initialization features
        self.initialization_desc = lambda: np.zeros((1, n_feats))
        ## Global construction of result
        if n_vals_i is not None:
            n_pert = self.features.k_perturb + 1
            ## Init global result
            self.initialization_output =\
                lambda: np.zeros((n_vals_i, n_feats, n_pert))
            ## Adding result
            self.add2result = self._defult_add2result
        else:
            ##### WARNING: Problem not solved
            ## Init global result
            self.initialization_output = lambda: []
            ## Adding result
            self.add2result = append_addresult_function

    def _format_map_vals_i(self, sp_typemodel):
        "Format mapper to indicate external val_i to aggregate result."
        if sp_typemodel is not None:
            map_vals_i = create_mapper_vals_i(self.features, sp_typemodel)
            self.features._maps_vals_i = map_vals_i


def get_individuals(neighs, dists, discretized):
    "Transform individual regions."
    logi = np.zeros(discretized.shape[0]).astype(bool)
    dists_i = np.zeros(discretized.shape[0])
    for i in range(len(neighs)):
        logi_i = (discretized == neighs[i]).ravel()
        logi = np.logical_or(logi, logi_i).ravel()
        dists_i[logi_i] = dists[i]
    neighs_i = np.where(logi)[0]
    dists_i = dists_i[logi]
    return neighs_i, dists_i

#    def compute_aggdescriptors(self, discretizor, regionretriever, locs):
#        """Compute aggregate region desctiptors for each region in order to
#        save computational power or to ease some computations.
#        WARNING: This function calls to compute_aggcharacs_i, which is not
#        required to be in the descriptor models object methods (there are some
#        of them which they cannot be descomposed in that way).
#
#        discretizor: Discretization object or numpy.ndarray
#            the discretization information of the location elements. It is
#            the object or the numpy.ndarray that will map an element id to
#            a collected elements or regions.
#        regionretriever: relative_pos object
#            region retriever which is able to retrieve neighbours regions from
#            a retriever type and a relation of regions object.
#        locs: numpy.ndarray
#            the location of the elements in the space.
#
#        Parameters
#        ----------
#        aggcharacs: numpy.ndarray
#            the aggcharacs for each region or collection of elements.
#
#        """
#        if type(discretizor) == np.ndarray:
#            discretized = discretizor
#        else:
#            discretized = discretizor.discretize(locs)
#            discretized = discretized.reshape((discretized.shape[0], 1))
#        u_regs = np.unique(discretized)
#        null_values = self.initialization_desc()
#        aggcharacs = np.vstack([null_values for i in xrange(u_regs.shape[0])])
#        for i in xrange(u_regs.shape[0]):
#            reg = u_regs[i]
#            neighs, dists = regionretriever.retrieve_neighs(np.array([reg]))
#            neighs_i, dists_i = get_individuals(neighs, dists, discretized)
#            #locs_i = locs[list(neighs_i), :]
#            ## Que hacer con locs i?
#            if len(neighs_i) != 0:
#                aggcharacs[i, :] = self.compute_aggcharacs_i(neighs_i, dists_i)
#        return aggcharacs, u_regs, null_values

#    def compute_descriptors(self, i, predescriptors):
#        """Function to made last corrections for the descriptors computed
#        initially using neighbourhood information, by using a global
#        information.
#        """
#        return predescriptors

    ############
#    def generate_aggregation(self, retrievers):
#        "Probably in spatial descriptors models."
#        n = len(retrievers)
#        aggdescriptors = [None]*n
#        for i in range(n):
#            if retrievers[i].typeret == 'region':
#                aggdescriptors[i] = self.compute_aggdescriptor(retrievers[i])
#        return aggdescriptors

#    def retrieve_aggregation(self, agg_arr, feat_arr, reindices):
#        ## n_uuu is given by n_dim object parameter
#        u_v, uuu = np.unique(agg_arr), np.unique(feat_arr)
#        n_u, n_uuu, n_rein = u_v.shape[0], uuu.shape[0], reindices.shape[1]
#        res = np.zeros((n_u, n_uuu, n_rein)).astype(int)
#        for j in range(reindices.shape[1]):
#            for i in xrange(u_v.shape[0]):
#                logi = agg_arr == u_v[i]
#                logi = logi[reindices[:, j]]
#                feats = feat_arr[logi, :]
#                precharacs = self.compute_characs_i(feats)  # characs
#                #c = dict(Counter(feats[:, 0]))
#                #res[i, c.keys(), j] += np.array(c.values())
#                res[i, :, j] += characs

#    def compute_general_predescriptors(self, ):
#        pass

#    def compute_partial_descriptors(self, i, neighs_info, k, typefeats):
#        desc_i, desc_neighs = self._get_point_features(i, neighs_info, k,
#                                                       typefeats)
#        descriptors = self._compute_descriptors_spec(i, neighs, desc_i,
#                                                     desc_neigh)
#        return descriptors
