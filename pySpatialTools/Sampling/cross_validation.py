
"""
Cross-validation
----------------
This module includes and groups utilities for cross-validation and
performance evaluation for specific spatial use.

"""

import numpy as np
from sklearn.cross_validation import _BaseKFold


class QuantileKFold(_BaseKFold):
    """Quantile K-Folds cross validation iterator.

    Provides train/test indices to split data in train test sets.

    This cross-validation object is a variation of KFold, which
    returns quantile splitted folds following a given values.
    The folds are made by preserving an ordering of the values
    the percentage of samples for each class.

    Each fold is then used a validation set once while the k - 1 remaining
    fold form the training set.

    Parameters
    ----------
    y : array-like, [n_samples]
        Samples to split in K folds.
    n_folds : int, default=3
        Number of folds. Must be at least 2.
    indices : boolean, optional (default True)
        Return train/test split as arrays of indices, rather than a boolean
        mask array. Integer indices are required when dealing with sparse
        matrices, since those cannot be indexed by boolean masks.

    Examples
    --------
    >>> X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
    >>> y = np.array([0.2, 0.7, 0.1, 0.9])
    >>> skf = cross_validation.QuantileKFold(y, n_folds=2)
    >>> len(skf)
    2
    >>> print(skf)
    QuantileKFold(n_samples=4, n_folds=2)
    >>> for train_index, test_index in skf:
    ...    print("TRAIN:", train_index, "TEST:", test_index)
    ...    X_train, X_test = X[train_index], X[test_index]
    ...    y_train, y_test = y[train_index], y[test_index]
    TRAIN: [0 2] TEST: [1 3]
    TRAIN: [1 3] TEST: [0 2]

    Notes
    -----
    All the folds have size trunc(n_samples / n_folds), the last one has the
    complementary.

    See also
    --------
    StratifiedKFold: take label information into account to avoid building
    folds with imbalanced class distributions (for binary or multiclass
    classification tasks).

    """

    def __init__(self, values, n_folds=3, indices=True, k=None):
        super(QuantileKFold, self).__init__(len(values), n_folds, indices, k)
        self.values = np.asarray(values)

    def _iter_test_indices(self):
        n_folds = self.n_folds
        idx = np.argsort(self.values)
        splits = [(len(idx)/n_folds)*i for i in range(n_folds)]
        splits.append(len(idx))
        for i in range(n_folds):
            yield idx[splits[i]:splits[i+1]]

    def __repr__(self):
        return '%s.%s(n_samples=%s, n_folds=%i)' % (
            self.__class__.__module__,
            self.__class__.__name__,
            len(self.values),
            self.n_folds,
        )

    def __len__(self):
        return self.n_folds
