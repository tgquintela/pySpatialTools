
"""
SpatioTemporal_descriptormodels
-------------------------------
Main class to the class of model descriptors. This class contains the main
functions and indications to compute the local descriptors.

"""

import numpy as np
from spatial_descriptormodels import SpatialDescriptorModel


class SpatioTemporalDescriptorModel:
    """The spatio-temporal descriptor model is an interface to compute
    spatio-temporal descriptors for a descriptor model processer mainly.
    It contains the utility classes of:
        * Spatial descriptor models: extract the descriptors from spatial data.
    Its main function in the process of computing descriptors from points is to
    manage the dealing with perturbation of the system for the sake of testing
    predictors and models.

    TODO:
    -----
    - Unification of input unique ids

    """

    def _initialization(self):
        ## Main classes
        self.spdesc_models = []
        ## Parameters useful
        self.n_inputs = [0]

    def __init__(self, spdesc_models):
        """Spatial descriptor model initialization.

        Parameters
        ----------
        spdesc_models: list
            the spatial descriptor models that instance stores.

        """
        self._initialization()
        self._format_spdesc(spdesc_models)

    ############################### Formatters ################################
    ###########################################################################
    def _format_spdesc(self, spdesc_models):
        """Format spatial descriptors.

        Parameters
        ----------
        spdesc_models: list
            the spatial descriptor models that instance stores.

        """
        ## 0. Parsing inputs
        if type(spdesc_models) != list:
            spdesc_models = [spdesc_models]

        ## 1. Creating spdesc
        spdescs = []
        for i in range(len(spdesc_models)):
            if type(spdesc_models[i]) == dict:
                spdescs.append(SpatialDescriptorModel(**spdesc_models[i]))
            elif type(spdesc_models[i]) == tuple:
                assert(len(spdesc_models[i]))
                spdesc_i = SpatialDescriptorModel(spdesc_models[i][0],
                                                  spdesc_models[i][1],
                                                  **spdesc_models[i][2])
                spdesc_models.append(spdesc_i)
            elif isinstance(spdesc_models[i], SpatialDescriptorModel):
                spdescs.append(spdesc_models[i])
        ## 2. Storing in instance
        self.spdesc_models = spdescs
        self.n_inputs = [self.spdesc_models[i].n_inputs
                         for i in range(len(self.spdesc_models))]

    def _format_inputs(self, indices, y=None):
        """Format input indices.

        Parameters
        ----------
        indices: np.ndarray or list
            the indices of the samples used to compute the model.
        y: np.ndarray or list
            the target we want to predict.

        Returns
        -------
        indices: np.ndarray or list
            the indices of the samples used to compute the model.
        y: np.ndarray or list
            the target we want to predict.

        """

        ## 0. Create indices formatting
        assert(np.max(indices) <= np.sum(self.n_inputs))
        ranges = np.cumsum([0]+self.n_inputs)
        spdesc_i, spdesc_k = -1*np.ones(len(indices)), -1*np.ones(len(indices))
        spdesc_i, spdesc_k = spdesc_i.astype(int), spdesc_k.astype(int)
        new_indices, new_y = [], []
        print indices
        print ranges
        for i in range(len(self.n_inputs)):
            logi = np.logical_and(indices >= ranges[i], indices <= ranges[i+1])
            spdesc_k[logi] = i
            spdesc_i[logi] = indices-ranges[i]
            new_indices.append(list(indices[logi]))
            if y is not None:
                new_y.append(y[logi])
        self._spdesc_i = spdesc_i.astype(int)
        self._spdesc_k = spdesc_k.astype(int)
        ## 1. Output format
        if y is None:
            return new_indices
        else:
            return new_indices, new_y

    def _format_output(self, y_pred):
        """Re-format the output in the same way the input is given.

        Parameters
        ----------
        y_pred: list
            the list of prediction targets for each possible spatial model.

        Return
        ------
        y_pred: np.ndarray
            the target predictions in the format is given the input.

        """
        n = sum([len(e) for e in y_pred])
        new_y_pred = np.zeros(n)
        for i in range(len(y_pred)):
            logi = self._spdesc_k == i
            assert(len(y_pred[i]) == np.sum(logi))
            new_y_pred[logi] = y_pred[i]
        return new_y_pred

    ########################## Main model functions ###########################
    ###########################################################################
    def fit(self, indices, y):
        """Use the SpatioTemporalDescriptorModel as a model.

        Parameters
        ----------
        indices: np.ndarray or list
            the indices of the samples used to compute the model.
        y: np.ndarray or list
            the target we want to predict.

        Returns
        -------
        self : returns an instance of self.

        """
        ## 0. Input check
        if type(indices) == list:
            assert(type(y) == list)
            assert(len(indices) == len(self.spdesc_models))
            assert(len(y) == len(self.spdesc_models))
        else:
            assert(type(indices) == np.ndarray)
            assert(type(y) == np.ndarray)
            assert(len(indices) == len(y))
        indices, y = self._format_inputs(indices, y)

        ## 1. Fit model
        for i in range(len(self.spdesc_models)):
            self.spdesc_models[i] = self.spdesc_models[i].fit(indices[i], y[i])
        return self

    def predict(self, indices):
        """Use the SpatioTemporalDescriptorModel as a model to predict targets
        from spatial descriptors.

        Parameters
        ----------
        indices: np.ndarray or list
            the indices of the samples used to compute the target predicted.

        Returns
        -------
        y_pred : np.ndarray
            the predicted target.

        """
        ## 0. Parse inputs
        if type(indices) == list:
            assert(len(indices) == len(self.spdesc_models))
        else:
            assert(type(indices) == np.ndarray)
            indices = self._format_inputs(indices)
            assert(len(indices) == len(self.spdesc_models))

        ## 1. Fit model
        y_pred = []
        for i in range(len(self.spdesc_models)):
            y_pred.append(self.spdesc_models[i].predict(indices[i]))

        ## 2. Format output
        y_pred = self._format_output(y_pred)

        return y_pred

    ########################## Interaction with data ##########################
    ###########################################################################
    def apply_aggregation(self, regs, agg_info, selectors):
        """Apply aggregations.

        Parameters
        ----------
        regs: np.ndarray
            the regions in which we want to aggregate information.
        agg_info: tuple
            the information to aggregate the information.
        selectors: int, tuple, np.ndarray
            how to select which retriever.

        """
        for i in range(regs):
            self.spdesc_models[i].apply_aggregation(regs[i], agg_info[i],
                                                    selectors[i])
