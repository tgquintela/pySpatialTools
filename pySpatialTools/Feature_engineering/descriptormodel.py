
"""
DescriptorModel
---------------

"""

import numpy as np


class DescriptorModel:
    """Abstract class for desctiptor models. Contain the common functions
    that are required for each desctiptor instantiated object.
    """

    def compute_aggdescriptors(self, discretizor, regionretriever, locs):
        """Compute aggregate region desctiptors for each region in order to
        save computational power or to ease some computations.

        discretizor: Discretization object or numpy.ndarray
            the discretization information of the location elements. It is
            the object or the numpy.ndarray that will map an element id to
            a collected elements or regions.
        regionretriever: relative_pos object
            region retriever which is able to retrieve neighbours regions from
            a retriever type and a relation of regions object.
        locs: numpy.ndarray
            the location of the elements in the space.

        Parameters
        ----------
        aggcharacs: numpy.ndarray
            the aggcharacs for each region or collection of elements.

        """
        if type(discretizor) == np.ndarray:
            discretized = discretizor
        else:
            discretized = discretizor.discretize(locs)
            discretized = discretized.reshape((discretized.shape[0], 1))
        u_regs = np.unique(discretized)
        null_values = self.initialization_desc()
        aggcharacs = np.vstack([null_values for i in xrange(u_regs.shape[0])])
        for i in xrange(u_regs.shape[0]):
            reg = u_regs[i]
            neighs, dists = regionretriever.retrieve_neighs(np.array([reg]))
            neighs_i, dists_i = get_individuals(neighs, dists, discretized)
            #locs_i = locs[list(neighs_i), :]
            ## Que hacer con locs i?
            if len(neighs_i) != 0:
                aggcharacs[i, :] = self.compute_aggcharacs_i(neighs_i, dists_i)
        return aggcharacs, u_regs, null_values


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
