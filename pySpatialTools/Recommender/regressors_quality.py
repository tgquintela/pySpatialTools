
"""
Regressors Recommenders
-----------------------
Recommenders based on the prediction of the quality measure assigned previously
over the known data.
Using the regressor selected we tried to create a model for predicting the
measure of quality in the out-of-sample data.

"""

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor,\
    ExtraTreesRegressor, AdaBoostRegressor
from sklearn.neighbors import RadiusNeighborsRegressor, KNeighborsRegressor


class RandomForestRecommender(SupervisedRmodel):
    """Recommender based on the Random Forests.
    
    Example
    -------
    from pySpatialTools.Recommender import RandomForestRecommender
    n, m, nt = 10000, 10, 20
    X = np.random.random((n, m))
    x_type = np.random.randint(0, nt, n)
    y = np.random.random(n)

    pars_model = {}
    cv = KFold
    pars_cv = {'n': n, 'n_folds': 3}
    recommender = RandomForestRecommender(pars_model, cv, pars_cv)
    model, measure = recommender.fit_model(X, x_type, y)
    Q = recommender.compute_quality(X, x_type)

    """
    name_desc = "Random Forest recommender"

    def __init__(self, model, pars_model):
        pass

    def retrieve_class_model(self):
        return RandomForestRegressor



class KneighRegRecommender(SupervisedRmodel):
    """Recommender based on the KNearest Neighbors regressor.
    
    Example
    -------
    from pySpatialTools.Recommender import KneighRegRecommender
    n, m, nt = 10000, 10, 20
    X = np.random.random((n, m))
    x_type = np.random.randint(0, nt, n)
    y = np.random.random(n)

    pars_model = {}
    cv = KFold
    pars_cv = {'n': n, 'n_folds': 3}
    recommender = KneighRegRecommender(pars_model, cv, pars_cv)
    model, measure = recommender.fit_model(X, x_type, y)
    Q = recommender.compute_quality(X, x_type)

    """
    name_desc = "K-NeighRegressor recommender"

    def retrieve_class_model(self):
        return KNeighborsRegressor


class RNeighRegRecommender(SupervisedRmodel):
    """Recommender based on the Nearest Neighbors by radius regressor.
    
    Example
    -------
    from pySpatialTools.Recommender import RNeighRegRecommender
    n, m, nt = 10000, 10, 20
    X = np.random.random((n, m))
    x_type = np.random.randint(0, nt, n)
    y = np.random.random(n)

    pars_model = {}
    cv = KFold
    pars_cv = {'n': n, 'n_folds': 3}
    recommender = RNeighRegRecommender(pars_model, cv, pars_cv)
    model, measure = recommender.fit_model(X, x_type, y)
    Q = recommender.compute_quality(X, x_type)

    """
    name_desc = "R-NeighRegressor recommender"

    def retrieve_class_model(self):
        return RadiusNeighborsRegressor


class GradientBoostingRecommender(SupervisedRmodel):
    """Recommender based on the Gradient Boosting regressor.
    
    Example
    -------
    from pySpatialTools.Recommender import GradientBoostingRecommender
    n, m, nt = 10000, 10, 20
    X = np.random.random((n, m))
    x_type = np.random.randint(0, nt, n)
    y = np.random.random(n)

    pars_model = {'loss': 'ls', 'learning_rate': 0.1, 'n_estimators': 100,
                  'max_depth': 3, 'max_features': None, 'alpha': 0.9}
    cv = KFold
    pars_cv = {'n': n, 'n_folds': 3}
    recommender = GradientBoostingRecommender(pars_model, cv, pars_cv)
    model, measure = recommender.fit_model(X, x_type, y)
    Q = recommender.compute_quality(X, x_type)

    """
    name_desc = "Gradient Boosting recommender"

    def retrieve_class_model(self):
        return GradientBoostingRegressor


class ExtraTreesRecommender(SupervisedRmodel):
    """Recommender based on the Extremal Trees regressor.
    
    Example
    -------
    from pySpatialTools.Recommender import ExtraTreesRecommender
    n, m, nt = 10000, 10, 20
    X = np.random.random((n, m))
    x_type = np.random.randint(0, nt, n)
    y = np.random.random(n)

    pars_model = {'n_estimators': 10, 'criterion': 'mse', 'max_depth': None,
                  'n_jobs'. 1}
    cv = KFold
    pars_cv = {'n': n, 'n_folds': 3}
    recommender = ExtraTreesRecommender(pars_model, cv, pars_cv)
    model, measure = recommender.fit_model(X, x_type, y)
    Q = recommender.compute_quality(X, x_type)

    """
    name_desc = "Extremal Trees recommender"

    def retrieve_class_model(self):
        return ExtraTreesRegressor


class AdaBoostRecommender(SupervisedRmodel):
    """Recommender based on the AdaBoost regressor.
    
    Example
    -------
    from pySpatialTools.Recommender import AdaBoostRecommender
    n, m, nt = 10000, 10, 20
    X = np.random.random((n, m))
    x_type = np.random.randint(0, nt, n)
    y = np.random.random(n)

    pars_model = {'base_estimator':None, 'n_estimators':50
                  'learning_rate':1.0, 'loss':'linear', 'random_state':None}
    cv = KFold
    pars_cv = {'n': n, 'n_folds': 3}
    recommender = AdaBoostRecommender(pars_model, cv, pars_cv)
    model, measure = recommender.fit_model(X, x_type, y)
    Q = recommender.compute_quality(X, x_type)

    """
    name_desc = "AdaBoost recommender"

    def retrieve_class_model(self):
        return AdaBoostRegressor

