
"""
Supervised Recommender Model
----------------------------
Module which contains the classes and needed functions for the computation of
recommender models based on supervised machine learning techniques.

TODO
----
The solution for different types is temporal.

"""

import numpy as np

from recommender_models import RecommenderModel
from aux_functions import fit_model, build_skmodel, build_cv, create_X

from sklearn.cross_validation import KFold


class SupervisedRmodel(RecommenderModel):
    "Generic abstract class for the supervised models for computing quality."
    modelclass = None
    model = None
    pars_model = {}
    cv = KFold
    pars_cv = {}

    def __init__(self, pars_model={}, cv=None, pars_cv={}):
        modelcl = self.retrieve_class_model()
        self.modelclass, self.pars_model = build_skmodel(modelcl, pars_model)
        self.cv, self.pars_cv = build_cv(cv, pars_cv)

    def fit_model(self, X, x_type, y):
        "Fit the model to predict y."
        ## 0. Prepare inputs
        cv = self.cv(**self.pars_cv)
        X = create_X(X, x_type, y)
        ## 1. Compute model and cross-validation performance measure
        model, measure = fit_model(self.modelclass, pars_model, cv, X, y)
        ## 2. Save selected model
        self.model = model
        return model, measure

    def compute_quality(self, X, x_type):
        "Compute the quality."
        ## 0. Prepare inputs
        X = create_X(X, x_type, y)
        ## 1. Infer quality
        Q = self.model.predict(X)
        return Q
