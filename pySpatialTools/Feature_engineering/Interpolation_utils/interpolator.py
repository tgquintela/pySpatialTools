
"""

TODO
----
- implementation of 3 cases:
    - autoretrieve
    - non-autoretrieve without features_i
    - non-autoretrieve with features_i
"""


class Interpolator:
    """
    TODO
    ----
    Matrix of weights between distances and own properties.
    (Weights not depending only on distances but also on ij type)
    Integrate special parameters in the functions with function-creators.

    """

    name_desc = "Density assignation descriptor"

    f_weights = None
    params_w = {}
    f_dens = None
    params_d = {}

    def __init__(self, f_weights, params_w, f_dens, params_d, feat_arr):
        """
        Parameters
        ----------
        f_weights: function
            function which eats dists and parameters and returns the weights
            associated with the related disctances.
        params_w: dict
            specific extra parameters of the f_weights
        f_dens: function
            function which eats weights, values and parameters and it returns
            the measure we want to compute.
        params_d: dict
            specific extra parameters of the f_dens.
        feat_arr: numpy.ndarray
            the feature array

        """

        self.f_weights = f_weights
        self.params_w = params_w
        self.f_dens = f_dens
        self.params_d = params_d

        self.features = feat_arr
        self.stype_model = stype_model   # Information about output

        ## Compulsary preset functions
        self.compute_value_i = lambda i: i
        self.to_complete_measure = lambda x: x

        self.compute_aggcharacs_i = lambda x:\
            x.mean(1).reshape((1, x.shape[1]))
        # todo: complete
        self.initialization_desc = lambda: np.zeros((1, n_feats))
        self.initialization_output = lambda x: np.zeros((nvals_i, n_feats, x))

    def compute_characs(self, i, neighs, dists):
        """Function for computing spatial descriptors from the characterizers
        and distances of its neighbours.
        """
        weights = self.f_weights(dists, **self.params_w)
        feat_i, feat_neighs = self.features[neighs], self.features[neighs]
        characs = self.f_dens(feat_i, feat_neighs, weights, **self.params_d)
        return characs
