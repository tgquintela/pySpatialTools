
"""
DistancesDescriptor
-------------------
This is a dummy descriptor for a phantom features. It uses the spatial relative
position of the neighbourhood information to compute descriptors.

"""

import numpy as np
from collections import Counter
from itertools import product
from pySpatialTools.FeaturesManagement import DescriptorModel


class DistanceDescriptors(DescriptorModel):
    """Distance descriptor models.
    """

    name_desc = "Distance descriptor"
    _nullvalue = 0

    def __init__(self, funct, regions, type_infeatures=None,
                 type_outfeatures=None):
        "The inputs are the needed to compute model_dim."
        ## Initial function set
        self._format_default_functions()
        self.set_functions(type_infeatures, type_outfeatures)
        self.set_global_info(funct, regions)
        ## Check descriptormodel
        self._checker_descriptormodel()

    def set_global_info(self, funct, regions):
        """Store the mapping regions to indices and the sum of coincidences
        matrix and the total possible combinations to be able to normalize
        later."""
        u_regs = np.unique(regions)
        n_reg = len(u_regs)

        def map_regions2idxs(reg, k):
            return np.where(regions[:, k] == reg)[0][0]
        self._map_regions2idxs = map_regions2idxs
        self._counting = np.zeros((n_reg, n_reg)).astype(int)
        self._function = funct

        ## Computations of total possible combinations
        total_combs = np.zeros((n_reg, n_reg)).astype(int)
        c = Counter(np.array([self._map_regions2idxs(e) for e in regions]))
        for p in product(*[c.keys(), c.keys()]):
            if p[0] != p[1]:
                total_combs[p[0], p[1]] = c[p[0]]*c[p[1]]
            else:
                total_combs[p[0], p[1]] = (c[p[0]]*(c[p[1]] - 1))/2
        self._totals_combinations = total_combs

    def compute(self, pointfeats, point_pos):
        """Compute the descriptors using the information available."""
        descriptors = distances_descriptors(pointfeats, point_pos,
                                            self._function)
        return descriptors

    def to_complete_measure(self, measure):
        """Main function to compute the complete normalized measure of pjensen
        from the matrix of estimated counts.
        """
        return measure

    def relative_descriptors(self, i, neighs_info, desc_i, desc_neigh, vals_i):
        """Completion of individual descriptor of neighbourhood by crossing the
        information of the individual descriptor of point to its neighbourhood
        descriptor.
        """
        descriptors = []
        for iss_i in range(len(desc_neigh)):
            descriptors.append(compute_loc_M_index(vals_i, desc_neigh[iss_i],
                               self.globals_))
        descriptors = np.array(descriptors)
        return descriptors


def transform_desc_neighs():
    pass


def distances_descriptors(pointfeats, point_pos, f):
    """Distances descriptors.
    """
    descriptors = []
    for k in range(len(pointfeats)):
        descriptors_k = []
        for i in range(len(pointfeats[k])):
            descriptors_ki = {}
            for nei in range(len(pointfeats[k][i])):
                descriptors_ki[pointfeats[k][i][nei]] = f(point_pos[k][i][nei])
            descriptors_k.append(descriptors_ki)
        descriptors.append(descriptors_k)
    return descriptors
