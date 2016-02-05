
"""
Aggcorr Model
-------------
Module which contains main functions to sequencially add features and the
required normalization.

"""


class SumAggCorrModel:
    """Class which contains the process of aggregate and compute measures
    sequencially.
    """

    def aggregate_characs(corr_loc_agg, corr_loc_i):
        """Function for sequencial aggregation of characterizers.

        TODO
        -----
        More generic function of aggregation
        """
        corr_loc_agg = corr_loc_agg + corr_loc_i
        return corr_loc_agg


class Pjensen_agg(SumAggCorrModel):

    def __init__(self):
        pass

    def normalize(corr_loc, N_t):
        "Normalization constant."
        pass
