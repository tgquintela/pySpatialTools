
"""
DescriptorModel
---------------
Module which contains the abstract class method for computing descriptor
models from puntual features.

"""

import numpy as np


class DescriptorModel:
    """Abstract class for desctiptor models. Contain the common functions
    that are required for each desctiptor instantiated object.
    """

    def complete_desc_i(self, i, neighs_info, desc_i, desc_neighs, vals_i):
        """Dummy completion for general abstract class."""
        ## TOTEST
        desc = []
        for k in xrange(len(vals_i)):
            desc.append(self.relative_descriptors(i, neighs_info, desc_i[k],
                                                  desc_neighs[k], vals_i[k]))
        descriptors = np.vstack(desc)
        return descriptors

    def relative_descriptors(self, i, neighs_info, desc_i, desc_neigh, vals_i):
        "General default relative descriptors."
        return desc_neigh

    ####################### Compulsary general functions ######################
    ###########################################################################

    ################# Dummy compulsary overwritable functions #################
    def to_complete_measure(self, corr_loc):
        """Main function to compute the complete normalized measure of pjensen
        from the matrix of estimated counts.
        """
        return corr_loc

#    def add2result(self, res_total, res_i):
#        """Addding results to the final aggregated result. We assume here
#        additivity property.
#        TODO: List append possibility
#        """
#        return res_total + res_i

    ###########################################################################
    ########################## Formatter functions ############################
    ###########################################################################


###############################################################################
###############################################################################
###############################################################################
###############################################################################
# TO RETRIEVER ??
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
