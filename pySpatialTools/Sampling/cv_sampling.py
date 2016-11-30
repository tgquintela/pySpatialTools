
"""
Cross-validation sampling
-------------------------
This module contains the functions used to crete a a cross validation
division for the data.


TODO
----
Create a class structure to create samplings
Create a class which is able to create a cv object
"""


class Categorical_Sampler:

    def __init__(self, points_cat, weights, precomputed=False,
                 repetition=False):
        if precomputed:
            len(points_cat) == len(weights)
        self.points_cat = points_cat
        self.weights = weights
        self.repetition = repetition

    def sample(self, n, fixed=True):
        if self.repetition and fixed:
            pass
        elif self.repetition and not fixed:
            pass
        elif not self.repetition and fixed:
            pass
        elif not self.repetition and not fixed:
            pass

    def generate_cv_sampling(self):
        pass


class Spatial_Sampler:

    def __init__(self, points_com, com_stats):
        self.points_com = points_com
        self.com_stats = com_stats
        self.n_com = len(self.com_stats)

    def sample(self, n):
        pass

    def retrieve_icom(self, icom):
        if type(self.points_com) == np.ndarray:
            indices = np.where(self.points_com == icom)[0]
        else:
            indices = np.zeros(len(self.points_com)).astype(bool)
            for i in xrange(len(self.points_com)):
                indices[i] = icom in self.points_com[i]
            indices = np.where(indices)[0]
        return indices

    def retrieve_non_icom(self, icom):
        if type(self.points_com) == np.ndarray:
            indices = np.where(self.points_com != icom)[0]
        else:
            indices = np.zeros(len(self.points_com)).astype(bool)
            for i in xrange(len(self.points_com)):
                indices[i] = icom not in self.points_com[i]
            indices = np.where(indices)[0]
        return indices

    def generate_cv_sampling(self):
        for icom in self.com_stats:
            r_type = self.retrieve_icom(icom)
            non_r_type = self.retrieve_non_icom(icom)
            yield r_type, non_r_type


class CV_sampler:
    """Sampler for creating CV partitions."""

    def __init__(self, f, m):
        pass

    def generate_cv(self):
        pass
