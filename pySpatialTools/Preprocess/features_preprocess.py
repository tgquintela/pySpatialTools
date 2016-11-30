
"""
"""

import numpy as np
from itertools import product


def combinatorial_combination_features(features):
    """Transform a categorical multidimension element feature matrix to one
    categorical feature.

    Parameters
    ----------
    features: np.ndarray, shape (n, m)
        the features we want to join. There are `m` categorical dimensions.

    Returns
    -------
    new_features: np.ndarray, shape (n, 1)
        the new features coded in only one integer dimension.
    translator: np.ndarray, shape (n, m)
        the ordered possible features combination.

    """
    assert(len(features.shape) == 2)
    features = features.astype(int)

    n_dim = features.shape[1]
    uniques_ = [tuple(np.unique(features[:, i])) for i in range(n_dim)]
    dims_ = [len(uniques_[i]) for i in range(n_dim)]
    pos_combinations = np.product(dims_)

    differenciator = np.array([uniques_[i][0] - 1 for i in range(n_dim)])
    translator = differenciator*np.ones((pos_combinations, n_dim))
    i = 0
    for p in product(*uniques_):
        translator[i] = np.array(p)
        i += 1
    assert(np.all(translator != differenciator))
    assert(len(translator) == pos_combinations)
    assert(i == pos_combinations)
    translator = translator.astype(int)

    new_features = -1*np.ones((len(features), 1))
    for i in xrange(pos_combinations):
        new_features[np.all(features == translator[i], axis=1)] = i
    assert(np.all(new_features != -1))
    new_features = new_features.astype(int)

    return new_features, translator
