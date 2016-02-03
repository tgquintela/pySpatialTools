
"""
FeaturesCollection
------------------
Module which contains features collections framework. It is useful to manage
data with heterogenous elements with different variables.
"""

import numpy as np
import pandas as pd


class FeaturesAggregated:
    "Class which stores fetures of heterogenous elements."

    def __init__(self, ):
        pass


class FeaturesCollection:
    """Class which stores features of heterogenous elements.
    """

    out = 'array'  # 'dict', 'tuple', 'list'

    def __init__(self, features_elements, features_names=None, out='array',
                 features_types=[]):
        """
        list of dicts
        list of objects
        pandas.dataframe
        numpy.ndarray

        continious, discrete
        """
        if type(features_elements) == list:
            if type(features_elements[0]) == dict:
                pass
            self.features = features_elements
        elif type(features_elements) == np.ndarray:
            self.features = features_elements
            self.features_names = features_names

        elif type(features_elements) == pd.DataFrame:
            self.features = features_elements.as_matrix()
            self.features_names = list(features_elements.columns)
        self.out = out

    @property
    def n_features(self):
        "Number of element features."
        return len(self.features_names)

    def __getitem__(self, i, varname=None):
        if isinstance(varname, slice):
            varname = self.features_names[varname]
        elif isinstance(varname, list):
            varname = [self.features_names[e] for e in varname]
        elif isinstance(varname, int):
            varname = [self.features_names[varname]]
        elif isinstance(varname, str):
            try:
                varname = self.features_names.index(varname)
            except:
                return np.array([])
        item = self.features[i, :]
        item = self.features[i]


        return item

    def __len__(self):
        return self.n_features

    def map_discrete_vars(self):
        return reverse_mapper
