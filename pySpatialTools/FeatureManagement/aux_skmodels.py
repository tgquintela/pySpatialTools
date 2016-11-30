
"""
skmodels
--------
The auxiliar scikit-based models.

"""

from sklearn.base import BaseEstimator


class DummySkmodel(BaseEstimator):
    """Dummy scikit-based model to complete the use of the spatial descriptors
    models as a models to predict targets.
    """

    def __init__(self, normalize=False):
        """

        Parameters
        ----------
        normalize: boolean
            if normalize the training data.

        """
        self.normalize = normalize

    def fit(self, X, y):
        """Fit model.

        Parameters
        ----------
        X: np.ndarray or scipy.sparse
            Training data
        y: np.ndarray
            the target we want to predict.

        Returns
        -------
        self : returns an instance of self

        """
        assert(len(X.ravel()) == len(y))
        self._rescale = (y.max()-y.min())/(X.max()-X.min()), y.min()
        return self

    def predict(self, X):
        """Predict values from model.

        Parameters
        ----------
        X: np.ndarray or scipy.sparse
            Samples data.

        Returns
        -------
        y_pred: np.ndarray; shape, (n_samples,)
            Returns predicted values.

        """
        if self.normalize:
            y_pred = (X.ravel()-X.min())*self._rescale[0]+self._rescale[1]
        else:
            y_pred = X.ravel()
        return y_pred
