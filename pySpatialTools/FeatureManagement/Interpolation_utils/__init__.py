
"""
Interpolation
=============
Module which contains the functions to spatially interpolate features.

TODO
----
- Join both ways into 1.

"""

## Interpolation
from general_interpolation import general_interpolate

## Density assignation
from density_assignation import general_density_assignation
from density_assignation_process import DensityAssign_Process




## Creation function descriptors

##        """
##        Parameters
##        ----------
##        f_weights: function
##            function which eats dists and parameters and returns the weights
##            associated with the related disctances.
##        params_w: dict
##            specific extra parameters of the f_weights
##        f_desc: function
##            function which eats weights, values and parameters and it returns
##            the measure we want to compute.
##        params_d: dict
##            specific extra parameters of the f_desc.
##
##        """
###     feat_arr, f_weights, f_desc, params_d={}, params_w={},
###        mapper_vals_i=None, sp_typemodel='matrix', f_add2retult=None,
###        compute_aggcharacs_i=None, to_complete_measure=None,
###                 name_desc=""):
##        # TODO: Transformation of point_pos?
##        ## Functions to compute descriptors
##        self.f_weights = lambda x: f_weights(x, **params_w)
##        self.f_desc = lambda x, y: f_desc(x, y, **params_d)
##    def compute_predescriptors(self, i, neighs, dists, reindices, k):
##        """
##        """
##        i, neighs = reindices[i, k], reindices[neighs, k]
##        weights = self.f_weights(dists)
##        feat_i, feat_neighs = self.features[neighs], self.features[neighs]
##        characs = self.f_dens(feat_i, feat_neighs, weights, **self.params_d)
##        return characs
