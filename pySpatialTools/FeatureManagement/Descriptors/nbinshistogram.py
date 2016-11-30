
"""
NBinsHistogram
--------------
Descriptor which counts a histogram by using nbins.
"""

import numpy as np
from descriptormodel import BaseDescriptorModel

## Specific functions
from ..aux_descriptormodels import\
    characterizer_1sh_counter, sum_reducer, null_completer,\
    aggregator_1sh_counter, sum_addresult_function, counter_featurenames,\
    count_out_formatter_general, null_out_formatter,\
    count_out_formatter_dict2array


class NBinsHistogramDesc(BaseDescriptorModel):
    """Model of spatial descriptor computing by binning and counting the type
    of the neighs represented in feat_arr.
    WARNING: Probably it is more efficient to binning first the all the feature
    data and after apply only counting.

    """

    name_desc = "N-bins histogram descriptor"
    _nullvalue = 0

    def __init__(self, n_bins, features=None, type_infeatures=None,
                 type_outfeatures=None):
        """The inputs are the needed to compute model_dim.

        Parameters
        ----------
        n_bins: int
            the number of bins we are going to use in order to make the
            histogram.
        features: np.ndarray
            the features in a array_like mode.
        type_infeatures: str, optional (default=None)
            type of the input features.
        type_outfeatures: str, optional (default=None)
            type of the output features.

        """
        ## Global initialization
        self.default_initialization()
        ## Initial function set
        self.selfdriven = False
        self._format_default_functions()
        self.set_functions(type_infeatures, type_outfeatures)
        ## Check descriptormodel
        self._assert_correctness()
        ## Globals initialization
        self.globals_ = [n_bins, None, None, False]
        if features is not None:
            self.set_global_info(features)

    ###########################################################################
    ####################### Compulsary main functions #########################
    ###########################################################################
    def compute(self, pointfeats, point_pos):
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

    ###########################################################################
    ############################# Extra functions #############################
    ###########################################################################
    def transform_features(self, features):
        """Transform the features into a discretize version of them in order
        to make counting or apply other descriptormodel.

        Parameters
        ----------
        features: np.ndarray
            the features in a array_like mode.

        """
        if self.globals_[2] is not None:
            features = self.globals_[2](features)
        return features

    ###########################################################################
    ##################### Non-compulsary main functions #######################
    ###########################################################################
    def set_global_info(self, features, transform=True):
        """Set global information for future tasks. It sees all the available
        features and compute some interesting quantities in order to be used
        during the descriptormodel computation.

        Parameters
        ----------
        features: np.ndarray
            the features in a array_like mode.
        transform: boolean (default=True)
            if we want to return transformed features.

        """
        self.globals_[0]
        mini, maxi = features.min(), features.max()
        diff = (maxi - mini)/float(self.globals_[0])
        borders = [mini+diff*i for i in range(1, self.globals_[0])]
        borders = borders + [maxi]
        self.globals_[1] = borders

        def binning(feats):
            """Binning function.

            Parameters
            ----------
            features: np.ndarray
                the features in a array_like mode.

            Returns
            -------
            binned_feats: np.ndarray
                the features after binning. Discretized features.

            """
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

        Parameters
        ----------
        type_infeatures: str, optional (default=None)
            type of the input features.
        type_outfeatures: str, optional (default=None)
            type of the output features.

        """
        assert(type_infeatures in [None, 'ndarray'])
        if type_outfeatures is None:
            self._out_formatter = count_out_formatter_general
        elif type_outfeatures == 'dict':
            self._out_formatter = null_out_formatter
        else:
            self._out_formatter = count_out_formatter_dict2array


class HistogramDistDescriptor(BaseDescriptorModel):
    """Descriptor which creates a histogram of distances.
    """
    name_desc = "Histogram of Distances"
    _nullvalue = 0

    def __init__(self, start, stop, n_points, ks, logscale=False):
        """The inputs are the needed to compute model_dim.

        Parameters
        ----------
        start: float
            the start point of the histogram. The minimum value.
        stop: float
            the stop point of the histogram. The maximum value.
        n_points: int
            the number of intervals of the histogram.
        ks: np.ndarray
            quantity of histograms.
        logscale: boolean
            if logarithmic scale.

        """
        ## Global initialization
        self.default_initialization()
        ## Initial function set
        self.selfdriven = False
        self._format_default_functions()
        self._set_parameters(start, stop, n_points, logscale, ks)
        ## Check descriptormodel
        self._assert_correctness()

    def _format_default_functions(self):
        self._out_formatter = null_out_formatter

    def compute(self, pointfeats, point_pos):
        """It compute a histogram over the positions.

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
        n_bins, n_ks = len(self._globals[0])-1, len(self._globals[1])
        mult = np.arange(n_ks)*n_bins
        descriptors = np.zeros((len(pointfeats), n_bins*n_ks))
        for i in range(len(descriptors)):
            for j in range(len(point_pos[i])):
                logi = j < self._globals[1]
                for k in range(n_bins):
                    bin_b = self._globals[0][k+1] >= point_pos[i][j]
                    bin_b = bin_b and self._globals[0][k] <= point_pos[i][j]
                    if bin_b:
                        descriptors[i][mult[logi]+k] += 1
                        break
        return descriptors

    def _set_parameters(self, start, stop, n_points, logscale, ks):
        """Setting parameters to the method.

        Parameters
        ----------
        start: float
            the start point of the histogram. The minimum value.
        stop: float
            the stop point of the histogram. The maximum value.
        n_points: int
            the number of intervals of the histogram.
        logscale: boolean
            if logarithmic scale.
        ks: np.ndarray
            quantity of histograms.

        """
        bins = create_binningdist(start, stop, n_points, logscale)
        ks = np.array([ks]) if type(ks) else np.array(ks)
        self._globals = bins, ks.ravel()
        self._f_default_names = lambda x: range(len(ks)*len(bins))


def create_binningdist(start, stop, n_points, logscale=True):
    """Create binning of distances.

    Parameters
    ----------
    start: float
        the start point of the histogram. The minimum value.
    stop: float
        the stop point of the histogram. The maximum value.
    n_points: int
        the number of intervals of the histogram.
    logscale: boolean
        if logarithmic scale.

    Returns
    -------
    bins: np.ndarray
        the bins definitions.

    """
    if start < 1:
        bins = np.logspace(0, np.log10(stop+1), n_points+1)-1
    else:
        bins = np.logspace(np.log10(start), np.log10(stop), n_points+1)
    return bins
