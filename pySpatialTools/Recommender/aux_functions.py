
"""
auxiliar functions
------------------
Functions which complement the tasks of the other modules.

"""

import numpy as np
from sklearn.cross_validation import KFold, LeaveOneOut, LeaveOneLabelOut,\
    LeavePOut, StratifiedKFold


def fit_model(model, pars_model, cv, X, y):
    """Function to fit the model we want.

    Parameters
    ----------
    model: model object
        an object which represents a model and has the functions class fit and
        score.
    pars_model: dict
        the parameters of the selected model.
    cv: cross-validation object or a list of tuples of indices
        the cross validation object.
    X: numpy.ndarray, shape (n, m)
        the data matrix.
    y: numpy.ndarray, shape (n,)
        the labels we want to predict with the model.

    Return
    ------
    model: model object
        an object which represents a model.
    measure: float
        the performance of the model in the cv-test.

    """
    models, measures = [], []
    for train_index, test_index in cv:
        ## Extract Kfold
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        ## Fit models
        model_i = model(**pars_model)
        models.append(model_i.fit(X_train, y_train))
        ## Compute measure
        measures.append(model_i.score(X_test, y_test))

    i = np.argmax(measures)
    model, measure = models[i], measures[i]
    return model, measure


## Auxiliary functions to normalize inputs
###############################################
def build_skmodel(model, pars_model):
    """Return an sklearn model object.

    Parameters
    ----------
    model: non-instantiated model, str
        model we want to apply in order to model the data.
    pars_model: dict
        paramters of the model.

    Returns
    -------
    model: non-instantiated model
        the instantiated model.
    pars_model: dict
        paramters of the model.

    """
    model, pars_model = model, pars_model

    return model, pars_model


def build_cv(cv, pars_cv):
    """Return something valid as Cross-validation sklearn object.
    """
    if cv is None:
        cv = KFold(**pars_cv)
    return cv


def create_X(X, x_type, y):
    """
    """
    x_type = x_type.reshape(x_type.shape[0], 1)
    X = np.concatenate([X, x_type], axis=1)
    return X
