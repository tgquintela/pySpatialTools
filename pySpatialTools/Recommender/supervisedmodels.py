
"""
Supervised Recommender Model
----------------------------
Module which contains the classes and needed functions for the computation of
recommender models based on supervised machine learning techniques.

TODO
----
The solution for different types is temporal.
Instantiate the models before?

"""

import numpy as np

from recommender_models import RecommenderModel
from aux_functions import fit_model, build_skmodel, build_cv, create_X

from sklearn.cross_validation import KFold


class SupervisedRmodel(RecommenderModel):
    "Generic abstract class for the supervised models for computing quality."

    modelclass = None
    model = None
    measure = 0.
    pars_model = {}
    cv = KFold
    pars_cv = {}

    def __init__(self, modelcl, pars_model={}, cv=None, pars_cv={}):
        """
        Parameters
        ----------
        modelcl: model object
            the model object.
        pars_model:
            the parameters needed to initilize the model object.
        cv: cross-validation object
            the
        pars_cv: dict
            the parameters needed to initilize the cross-validation class.

        """
        #modelcl = self.retrieve_class_model()
        self.modelclass, self.pars_model = build_skmodel(modelcl, pars_model)
        self.cv = build_cv(cv, pars_cv)

    def fit_model(self, X, x_type, y):
        """Fit the model to predict y.

        Parameters
        ----------
        X: numpy.ndarray, shape (n, m)
            the data matrix.
        x_type: numpy.ndarray, shape (n,)
            an array which represents the categorical variable of a location.
        y: numpy.ndarray, shape(n,)
            the label array.

        Returns
        -------
        model: model class
            a class which is able to fit data and has the functions class fit
            and score.
        measure: float
            the performance of the model in the cv-test.

        """
        ## 0. Prepare inputs
        ## 1. Compute model and cross-validation performance measure
        model, measure = fit_model(self.modelclass, self.pars_model, self.cv,
                                   X, y)
        ## 2. Save selected model
        self.model, self.measure = model, measure
        return model, measure

    def compute_quality(self, X, x_type):
        """Compute the quality. Supervised models try to predict out-of-sample
        some quality measure computed a priori.

        Parameters
        ----------
        X: numpy.ndarray, shape (n, m)
            the data matrix.
        x_type: numpy.ndarray, shape (n,)
            an array which represents the categorical variable of a location.

        Returns
        -------
        Q: numpy.ndarray, shape (n,)
            the quality prediction of the sample.

        """
        ## 0. Prepare inputs
        ## 1. Infer quality
        Q = self.model.predict(X)
        return Q

    def compute_kbest_type(self, X, y, kbest):
        """Compute the k best type and their quality.

        Parameters
        ----------
        X: numpy.ndarray, shape (n, m)
            the data matrix.
        x_type: numpy.ndarray, shape (n,)
            an array which represents the categorical variable of a location.
        kbest: int
            the number of best types we want to get.

        Returns
        -------
        Qs: numpy.ndarray, shape (N, kbest)
            the quality values for each sample.
        idxs: numpy.ndarray, shape (N, kbest)
            the indices of the k-best types for each sample.

        """

        return Qs, idxs
