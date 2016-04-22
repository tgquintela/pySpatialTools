
"""
NBinsHistogram
--------------
Descriptor which counts a histogram by using nbins.
"""

import numpy as np
from ..descriptormodel import DescriptorModel

## Specific functions
from ..aux_descriptormodels import\
    characterizer_1sh_counter, sum_reducer, null_completer,\
    aggregator_1sh_counter, sum_addresult_function, counter_featurenames,\
    count_out_formatter_general, null_out_formatter,\
    count_out_formatter_dict2array


class NBinsHistogramDesc(DescriptorModel):
    """Model of spatial descriptor computing by binning and counting the type
    of the neighs represented in feat_arr.
    WARNING: Probably it is more efficient to binning first the all the feature
    data and after apply only counting.

    """

    name_desc = "N-bins histogram descriptor"
    _nullvalue = 0

    def __init__(self, n_bins, features=None, type_infeatures=None,
                 type_outfeatures=None):
        """The inputs are the needed to compute model_dim."""
        ## Initial function set
        self._format_default_functions()
        self.set_functions(type_infeatures, type_outfeatures)
        ## Check descriptormodel
        self._checker_descriptormodel()
        ## Globals initialization
        self.globals_ = [n_bins, None, None, False]
        if features is not None:
            self.set_global_info(features)

    ###########################################################################
    ####################### Compulsary main functions #########################
    ###########################################################################
    def compute_characs(self, pointfeats, point_pos):
        """Compulsary function to pass for the feture retriever.

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
        if not self.globals_[3]:
            if type(pointfeats) != np.ndarray:
                pointfeats = self.transform_features(pointfeats)[0]
            else:
                pointfeats = self.transform_features(pointfeats)
        descriptors = characterizer_1sh_counter(pointfeats, point_pos)
        ## TODO: Transform dict to array and reverse
        #keys = [self.mapper[key] for key in counts.keys()]
        #descriptors[0, keys] = counts.values()
        return descriptors

    def reducer(self, aggdescriptors_idxs, point_aggpos):
        """Reducer gets the aggdescriptors of the neighbourhood regions
        aggregated and collapse all of them to compute the descriptor
        associated to a retrieved neighbourhood.
        """
        descriptors = sum_reducer(aggdescriptors_idxs, point_aggpos)
        return descriptors

    def aggdescriptor(self, pointfeats, point_pos):
        "This function assigns descriptors to a aggregation unit."
        descriptors = self.compute_characs(pointfeats, point_pos)
        return descriptors

    ###########################################################################
    ############################# Extra functions #############################
    ###########################################################################
    def transform_features(self, features):
        if self.globals_[2] is not None:
            features = self.globals_[2](features)
        return features

    ###########################################################################
    ##################### Non-compulsary main functions #######################
    ###########################################################################
    def set_global_info(self, features, transform=True):
        self.globals_[0]
        mini, maxi = features.min(), features.max()
        diff = (maxi - mini)/float(self.globals_[0])
        borders = [mini+diff*i for i in range(1, self.globals_[0])]
        borders = borders + [maxi]
        self.globals_[1] = borders

        def binning(feats):
            binned_feats = -1*np.ones(feats.shape)
            for i in range(len(borders)):
                j = len(borders)-1-i
                binned_feats[features <= borders[j]] = j
            assert((binned_feats == -1).sum() == 0)
            binned_feats = binned_feats.astype(int)
            return binned_feats

        self.globals_[2] = binning

        if transform:
            self.globals_[3] = True
            return self.transform_features(features)

    def _format_default_functions(self):
        """Format default mutable functions."""
        self._out_formatter = count_out_formatter_general
        self._f_default_names = counter_featurenames
#        self._defult_add2result = sum_addresult_function

    def set_functions(self, type_infeatures, type_outfeatures):
        """Set specific functions knowing a constant input and output desired.
        """
        if type_outfeatures == 'dict':
            self._out_formatter = null_out_formatter
        else:
            self._out_formatter = count_out_formatter_dict2array
