
"""
DistancesDescriptor
-------------------
This is a dummy descriptor for a phantom features. It uses the spatial relative
position of the neighbourhood information to compute descriptors.

"""

import numpy as np
from collections import Counter
from itertools import product
from descriptormodel import BaseDescriptorModel
from ..aux_descriptormodels.out_formatters import null_out_formatter


class DistancesDescriptor(BaseDescriptorModel):
    """Dummy distances descriptor model created for testing purposes."""

    name_desc = "Dummy distances descriptor"

    def _initialization(self):
        ## Global initialization
        self.default_initialization()
        ## Local initialization
        self.selfdriven = False
        self._map2idx = lambda idx: idx
        self._funct = lambda x: x if x is not None else 0
        ## General compulsary functions
        self._out_formatter = null_out_formatter
        self._f_default_names = lambda x: range(x[1])

    def __init__(self, nfeats, map_idx=None, funct=None):
        """A dummy distance descriptor.

        Parameters
        ----------
        nfeats: int
            the number of bins or descriptor variables.
        map_idx: function
            a function which transforms the position to an index, usually a
            binning function.
        funct: function
            the function which transforms the position information into the
            descriptors 1 by 1.

        """
        ## Specific class parameters
        self._initialization()
        self._setting_information(nfeats, map_idx, funct)
        ## Check descriptormodel
        self._assert_correctness()

    def compute(self, pointfeats, point_pos):
        """From [iss][nei][feats] to [iss][feats]

        Parameters
        ----------
        pointfeats: list of arrays, np.ndarray
            the point features information. [iss][nei][feats]
        point_pos: list of arrays or np.ndarray.
            the element relative position of the neighbourhood.
            [iss][nei][rel_pos]

        Returns
        -------
        descriptors: list of arrays or np.ndarray or list of dicts
            the descriptor of the neighbourhood. [iss][feats]

        """
        return self._core_characterizer(pointfeats, point_pos)

    def set_functions(self, type_infeatures, type_outfeatures):
        """Dummy set for setting specific inputs and outputs.

        Parameters
        ----------
        type_infeatures: str, optional
            type of the input features.
        type_outfeatures: str, optional
            type of the output features.

        """
        if type_outfeatures == 'dict':
            self._core_characterizer = self._distances_characterizer_dict
        else:
            self._core_characterizer = self._distances_characterizer_array

    def _setting_information(self, nfeats, map_idx, funct):
        """Setting basic information for this class descriptormodel.

        Parameters
        ----------
        nfeats: int
            the number of bins or descriptor variables.
        map_idx: function
            a function which transforms the position to an index, usually a
            binning function.
        funct: function
            the function which transforms the position information into the
            descriptors 1 by 1.

        """
        self._n = nfeats
        if map_idx is not None:
            self._map2idx = map_idx
        if funct is not None:
            self._funct = funct

    def _distances_characterizer_dict(self, pointfeats, point_pos):
        """Core characterizer for general distance descriptor model. That
        function is used to output as a dict-form.

        Parameters
        ----------
        pointfeats: list of arrays, np.ndarray
            the point features information. [iss][nei][feats]
        point_pos: list of arrays or np.ndarray.
            the element relative position of the neighbourhood.
            [iss][nei][rel_pos]

        Returns
        -------
        descriptors: list of dicts
            the descriptor of the neighbourhood. [iss][feats]

        """
        descriptors = []
        if point_pos is None:
            return descriptors
        for iss_i in range(len(pointfeats)):
            descriptors_i = {}
            if point_pos[iss_i] is None:
                descriptors.append(descriptors_i)
                continue
            for nei in range(len(pointfeats[iss_i])):
                descriptors_i[self._map2idx(pointfeats[iss_i][nei])] =\
                    self._funct(point_pos[iss_i][nei])
            descriptors.append(descriptors_i)
        return descriptors

    def _distances_characterizer_array(self, pointfeats, point_pos):
        """Core characterizer for general distance descriptor model. That
        function is used to output as a array-form.

        Parameters
        ----------
        pointfeats: list of arrays, np.ndarray
            the point features information. [iss][nei][feats]
        point_pos: list of arrays or np.ndarray.
            the element relative position of the neighbourhood.
            [iss][nei][rel_pos]

        Returns
        -------
        descriptors: np.ndarray
            the descriptor of the neighbourhood. [iss][feats]

        """
        descriptors = np.ones((len(pointfeats), self._n))*self._funct(0)
        if point_pos is None:
            return descriptors
        for iss_i in range(len(pointfeats)):
            if point_pos[iss_i] is None:
                continue
            for nei in range(len(pointfeats[iss_i])):
                descriptors[iss_i][self._map2idx(pointfeats[iss_i][nei])] =\
                    self._funct(point_pos[iss_i][nei])
        return descriptors


class NormalizedDistanceDescriptor(DistancesDescriptor):
    """Distance descriptor models.
    """

    name_desc = "Normalized distance descriptor"
    _nullvalue = 0

    def __init__(self, regions, nfeats, map_idx=None, funct=None, k_perturb=0):
        """The normalized distance descriptormodel.

        Parameters
        ----------
        regions: np.ndarray
            the regions assigned. There is one column for each perturbation.
        nfeats: int
            the number of bins or descriptor variables.
        map_idx: function
            a function which transforms the position to an index, usually a
            binning function.
        funct: function
            the function which transforms the position information into the
            descriptors 1 by 1.
        k_perturb: int
            the number of perturbations.

        """
        ## Specific class parameters
        self._initialization()
        self._setting_information(nfeats, map_idx, funct)
        self.set_global_info(regions, nfeats, k_perturb)
        ## Check descriptormodel
        self._assert_correctness()

    def set_global_info(self, regions, n_reg, k_perturb):
        """Store the mapping regions to indices and the sum of coincidences
        matrix and the total possible combinations to be able to normalize
        later.

        Parameters
        ----------
        regions: np.ndarray
            the regions assigned. There is one column for each perturbation.
        n_reg: int
            the number of possible regions.
        k_perturb: int
            the number of perturbations.

        """
        self._counting = np.zeros((n_reg, n_reg, k_perturb+1)).astype(int)
        ## Computations of total possible combinations
        if len(regions.shape) == 2:
            if regions.shape[1] == 1 and regions.shape[1] != k_perturb+1:
                regions = regions.ravel()
            else:
                assert(regions.shape[1] == k_perturb+1)
                total_combs = np.zeros((n_reg, n_reg, k_perturb+1)).astype(int)
                for k in range(k_perturb+1):
                    c = Counter(np.array([self._map2idx(e)
                                          for e in regions[:, k]]))
                    total_combs[:, :, k] = compute_matrix_total_combs(c, n_reg)
                    self._totals_combinations = total_combs
        if len(regions.shape) == 1:
            c = Counter(np.array([self._map2idx(e) for e in regions]))
            total_combs = compute_matrix_total_combs(c, n_reg)
            if k_perturb == 0:
                self._totals_combinations =\
                    total_combs.reshape((n_reg, n_reg, 1))
            else:
                total_combs_aux = np.zeros((n_reg, n_reg, k_perturb+1))
                for k in range(k_perturb+1):
                    total_combs_aux[:, :, k] = total_combs
                self._totals_combinations = total_combs_aux

    def compute(self, pointfeats, point_pos):
        """Compute the descriptors using the information available.

        Parameters
        ----------
        pointfeats: list of arrays, np.ndarray or list of list of dicts
            the point features information. [iss][nei][feats]
        point_pos: list of arrays or np.ndarray.
            the element relative position of the neighbourhood.
            [iss][nei][rel_pos]

        Returns
        -------
        descriptors: list of arrays or np.ndarray or list of dicts
            the descriptor of the neighbourhood. [iss][feats]


        """
        descriptors = self._core_characterizer(pointfeats, point_pos)
        return descriptors

    def to_complete_measure(self, measure):
        """Main function to compute the complete normalized measure of pjensen
        from the matrix of estimated counts.

        Parameters
        ----------
        measure: np.ndarray
            the measure computed by the whole spatial descriptor model.

        Returns
        -------
        measure: np.ndarray
            the transformed measure computed by the whole spatial descriptor
            model.

        """
        assert(type(measure) == np.ndarray)
        assert(measure.shape == self._counting.shape)
        measure = np.multiply(measure, np.divide(self._totals_combinations,
                                                 np.multiply(self._counting,
                                                             self._counting)))
        return measure

    def complete_desc_i(self, i, neighs_info, desc_i, desc_neighs, vals_i):
        """Completion dont affect the descriptors neighbourhood computed, only
        the internal counting of coincidences of the class.

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

        Returns
        -------
        desc: list
            the descriptors for each perturbation and element.

        """
        self._counting_array(desc_neighs, vals_i)
        return desc_neighs

    def _counting_array(self, descriptors, vals_i):
        """Counting in a array.

        Parameters
        ----------
        descriptors: np.ndarray or list of dict
            the descriptors associated to the neighbourhood elements of each
            `iss` for each perturbation `k`.
        vals_i: list or np.ndarray
            the storable index information for each perturbation `k`.

        Returns
        -------
        desc: list
            the descriptors for each perturbation and element.

        """
        for k in range(len(descriptors)):
            c = Counter(descriptors[k].nonzero()[1])
            for v in c:
                self._counting[vals_i[k], v, k] += c[v]
        return descriptors


def compute_matrix_total_combs(c, n_reg):
    """Computing matrix of total combinations."""
    total_combs = np.zeros((n_reg, n_reg)).astype(int)
    for p in product(*[c.keys(), c.keys()]):
        if p[0] != p[1]:
            total_combs[p[0], p[1]] = c[p[0]]*c[p[1]]
        else:
            total_combs[p[0], p[1]] = (c[p[0]]*(c[p[1]] - 1))/2
    return total_combs
