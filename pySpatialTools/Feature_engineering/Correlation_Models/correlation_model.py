
"""
Correlation Model
-----------------
Main class to the class of correlation model. This class contains the main
functions, classes and indications to compute a correlation of features given
a distribution of features in space.

"""


class CorrelationModel:
    "General class for correlation models."
    model_dim = (0, 0)  # check compute_descriptors

    def __init__(self, descriptmodel, aggfeatmodel):
        self.descriptmodel = descriptmodel
        self.aggfeatmodel = aggfeatmodel

    def get_characterizers(self, i, k, feat_arr, point_i, reindices,
                           retriever, info_ret, cond_agg):
        """Function for getting characterizers from the spatial distribution of
        features.
        """
        val_i, characs =\
            self.descriptmodel.get_characterizers(i, k, feat_arr, point_i,
                                                  reindices, retriever,
                                                  info_ret, cond_agg)
        return val_i, characs

    def to_complete_measure(self, corr_loc, N_t):
        """Normalization of the measure of correlation.
        """
        corr_loc = self.aggfeatmodel.normalize(corr_loc, N_t)
        return corr_loc

    def aggregate_characs(self, corr_loc_agg, corr_loc_i):
        """Function for sequencial aggregation of characterizers.

        TODO
        -----
        More generic function of aggregation
        """
        corr_loc_agg = self.aggfeatmodel.aggregate_characs(corr_loc_agg,
                                                           corr_loc_i)
        return corr_loc_agg
