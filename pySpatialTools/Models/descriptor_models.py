
"""
Descriptor Model
----------------
Main class to the class of model descriptors. This class contains the main
functions and indications to compute the local descritptors given the
neighbourhood.

"""


class DescriptorModel:
    "General class for descriptor models."
    model_dim = (0, 0)  # check compute_descriptors

    def get_characterizers(self, i, k, feat_arr, point_i, reindices,
                           retriever, info_ret, cond_agg):
        """Retrieve local characterizers for i element and k permutation. It
        returns the column index in the output matrix correlation (val_i) and
        trivial descriptors of the neighbourhood (vals). These values are used
        for the specific model function compute_descriptors.
        """
        # Retrieve neighs
        info_i, cond_i = info_ret[i], cond_agg[i]
        neighs, type_n = retriever.retrieve_neighs(point_i, cond_i, info_i)
        # Get vals
        val_i = self.compute_value_i(i, k, feat_arr, reindices)
        vals = self.compute_vals_nei(retriever.aggfeatures, feat_arr, neighs,
                                     reindices, k, type_n)
        # Get characterizers
        characs = self.integrate_vals(vals, type_n)

        return val_i, characs
