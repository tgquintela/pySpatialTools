
"""
Aggregation module contains the class and the function needed to aggregate
some features regarding some aggregation variable or indication.

TODO
----
- personal function of aggregation locations.
"""

import pandas as pd
import numpy as np

from aggregation_utils import compute_aggregate_counts,\
    average_position_by_aggarr


class Aggregator:
    """Object class to compute all the aggregation.
    """
    aggcharacs = None

    def __init__(self, regionretriever):
        self.regionretriever = regionretriever

    def compute_characterizors(self, reg=None):
        ## 0. Format inputs
        if reg is None:
            regs = regionretriever.u_regs
        else:
            if type(reg) == int:
                regs = [reg]
            elif type(reg) in [list, np.ndarray]:
                regs = reg
        ## 1. Compute characterizers
        characterizers = []
        for reg in regs:
            neighs, dists = self.regionretriever.retrieve_neighregregions(reg)
            neighs_i, dists_i = self.regionretriever.retrieve_neighreg_i(reg)
            feats_i = feats[neighs_i, :]  ## TODO revise
            characs = self.descriptor.compute_characs(feats_i, dists_i)
            characterizers.append(characs)
        characterizers = np.array(characterizers)
        return characterizers, regs

    def set_aggcharacs(self, descriptor, reindices):
        """
        """
        for k in xrange(reindices.shape[1]):
            characterizers = []
            for i in xrange(descriptor.features.shape[0]):
                neighs, dists = self.regionretriever_i(i)
                characs = descriptor.compute_characs(reindices[i, k],
                                                     reindices[neighs, k],
                                                     dists)
                characterizers.append(characs)
            self.aggcharacs.append(np.array(characterizers))

    def compute_aggcharacs(self, i, descriptor, reindices, k):
        if reindices is not None:
            i, neighs = reindices[i, k], reindices[neighs, k]
        neighs, dists = self.regionretriever_i(i)
        characs = descriptor.compute_characs(reindices[i, k],
                                             reindices[neighs, k], dists)
        return characs

    def get_characs(self, i, neighs, dists_info, k=0, reindices=None):
        if self.aggcharacs is not None:
            characs[k][]
        else:
            descriptor.compute_characs(i, neighs, dists_info)
        return characs

    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    def recursive_adding(self, characs):
        ch = characs[0, :]
        for i in range(1, characs.shape[0]):
            ch = self.adding(ch, characs[i, :])
        return ch


    ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

class AggRetriever:
    """
    """
    regions_id = []
    aggcharacs = []
    nullstate = []

    regionretriever = None

    def __init__(self, regionretriever, descriptor):
        self.regionretriever, self.descriptor = regionretriever, descriptor

    def compute_aggcharacs(self):
        self.regions_id, self.aggcharacs = 
        self.nullstate = 
        self.descriptor = None

    def retrieve_neigh_i(self, i):
        """
        """
        disc_i = self.discretize(i)
        neigh_reg, dists = self.regionretriever.get_neighregions(disc_i)
        return neigh_reg, dists

    def discretize(self, i):
        "Obtain code."
        disc_i = self.regionretriever.discretize(i)
        return disc_i

    def retrieve_neigh(self, disc_i):
        """
        """
        neigh_reg, dists = self.regionretriever.get_neighregions(disc_i)
        return neigh_reg, dists

    def get_aggcharacs(self, discs, k):
        """Get aggcharacs from precomputed aggcharacs.
        TODO: not precomputed
        """
        # Format properly
        discs = discs if type(discs) == int else list(discs)
        # Compute needed vars
        length = len(discs)
        ins = [i for i in range(length) if discs[i] in self.regions_id]
        discs_in = [discs[i] for i in ins]
        # Retrieve aggcharacs
        characs_in = self.aggcharacs[discs_in, :, k]
        # Compute characs matrix
        characs = np.zeros((length, characs_in.shape[1]))
        for i in range(length):
            if i in ins:
                characs[i, :] = characs_in[ins.index(i)]
            else:
                characs[i, :] = self.nullstate
        return characs


class Aggregator:
    """Aggregate features following the aggregation indications (aggregation
    variable and aggregation function).
    """

    def __init__(self, funct, n_dim):
        """It is needed the identification of the variables agg_var, feat_vars
        and loc_vars through a dictionary.

        Parameters
        ----------
        funct: function
            the function of aggregation we want to apply.
        """
        self.funct = funct
        self.n_dim = n_dim

    def retrieve_aggregation(self, agg_arr, feat_arr, reindices):
        """Main function for retrieving aggregation.

        Parameters
        ----------
        agg_arr: numpy.ndarray
            the aggregation array. Each instance is tagged with a number of the
            region of aggregation.
        feat_arr: numpy.ndarray
            the feature array.
        reindices: array_like, None
            the possible permutations of indices considered.

        Return
        ------
        aggfeatures: array_like or dict
            aggregation of features.

        """
        ## n_uuu is given by n_dim object parameter
        u_v, uuu = np.unique(agg_arr), np.unique(feat_arr)
        n_u, n_uuu, n_rein = u_v.shape[0], uuu.shape[0], reindices.shape[1]
        aggfeatures = np.zeros((n_u, self.n_dim, n_rein)).astype(int)
        for j in range(reindices.shape[1]):
            for i in xrange(u_v.shape[0]):
                logi = agg_arr == u_v[i]
                logi = logi[reindices[:, j]]
                feats = feat_arr[logi, :]
                precharacs = self.compute_characs_i(feats)
                aggfeatures[i, :, j] += characs
        return aggfeatures


#agg = np.random.randint(0, 500, 10000)
#feat_arr = np.random.randint(0, 250, (10000, 1))
#reindices = np.vstack([np.random.permutation(10000) for i in range(10)]).T
#
#t0 = time.time()
#res = histogram_feats(agg, feat_arr, reindices)
#print time.time()-t0
#
#t0 = time.time()
#res2 = aggregated_counts(agg, feat_arr, reindices)
#print time.time()-t0


def histogram_feats(agg_arr, feat_arr, reindices, f=None):
    u_v, uuu = np.unique(agg_arr), np.unique(feat_arr)
    n_u, n_uuu, n_rein = u_v.shape[0], uuu.shape[0], reindices.shape[1]
    res = np.zeros((n_u, n_uuu, n_rein)).astype(int)
    for j in range(reindices.shape[1]):
        for i in xrange(u_v.shape[0]):
            logi = agg_arr == u_v[i]
            logi = logi[reindices[:, j]]
            feats = feat_arr[logi, :]
            res[i, :, j] += np.array(c.values())
    return res


def histogram_feats_i(feats, n_vals):
    res = np.zeros(n_vals)
    c = dict(Counter(feats[:, 0]))
    res[c.keys()] = np.array(c.values())
    return res










class Aggregator:
    """Aggregate features following the aggregation indications (aggregation
    variable and aggregation function).
    """

    def __init__(self, typevars):
        """It is needed the identification of the variables agg_var, feat_vars
        and loc_vars through a dictionary.
        """
        self.typevars = format_typevars(typevars)

    def retrieve_aggregation(self, df, reindices=None, funct=None):
        """Main function for retrieving aggregation.

        Parameters
        ----------
        df: pd.DataFrame
            the data with the loc_vars, feat_vars and agg_var to aggregate.
        reindices: array_like, None
            the possible permutations of indices considered.
        funct: function
            the function of aggregation we want to apply.

        Return
        ------
        agglocs: 
            aggregation of locations.
        aggfeatures: array_like or dict
            aggregation of features.

        """

        ## Correct inputs
        #################
        locs = df[self.typevars['loc_vars']].as_matrix()
        feat_arr = df[self.typevars['feat_vars']].as_matrix()
        agg_arr = df[self.typevars['agg_var']].as_matrix()
        if reindices is None:
            N_t = locs.shape[0]
            reindices = np.array(range(N_t)).reshape((N_t, 1))
        if len(feat_arr.shape) == 1:
            feat_arr = feat_arr.reshape(feat_arr.shape[0], 1)
        ######################################################
        ## Compute agglocs
        agglocs = average_position_by_aggarr(locs, agg_arr)
        ## Compute aggfeatures
        aggfeatures = create_aggregation(agg_arr, feat_arr, reindices,
                                         self.typevars, funct)
        ## Format output
        agglocs = np.array(agglocs)
        ndim, N_t = len(agglocs.shape), agglocs.shape[0]
        agglocs = agglocs if ndim > 1 else agglocs.reshape((N_t, 1))
        return agglocs, aggfeatures


def create_aggregation(agg_arr, feat_arr, reindices, typevars=None,
                       funct=None):
    "Create aggregation."

    ## 0. Formatting inputs
    typevars = format_typevars(typevars, feats_dim=feat_arr.shape[1])
    feat_vars, agg_var = typevars['feat_vars'], typevars['agg_var']
    df1 = pd.DataFrame(agg_arr, columns=[agg_var])
    df2 = pd.DataFrame(feat_arr, columns=feat_vars)
    df = pd.concat([df1, df2], axis=1)

    ## 1. Use specific function or default aggregate counts
    if funct is None:
        agg_desc, _ = compute_aggregate_counts(df, agg_var, feat_vars,
                                               reindices)
        agg_desc = agg_desc[agg_desc.keys()[0]]
    else:
        agg_desc = funct(df, agg_var, feat_vars, reindices)
    return agg_desc


def format_typevars(typevars, locs_dim=None, feats_dim=None):
    "Check typevars."
    if typevars is None:
        typevars = {'agg_var': 'agg'}
        if locs_dim is not None:
            loc_vars = [chr(97+i) for i in range(locs_dim)]
            typevars['loc_vars'] = loc_vars
        if feats_dim is not None:
            feat_vars = [str(i) for i in range(feats_dim)]
            typevars['feat_vars'] = feat_vars
    if 'agg_var' not in typevars.keys():
        typevars['agg_var'] = None
    return typevars


def map_multivars2key(multi, vals=None):
    "Maps a multivariate discrete array to a integer."
    n_dim, N_t = len(multi.shape), multi.shape[0]
    if vals is None:
        vals = []
        for i in range(n_dim):
            aux = np.unique(multi[:, i])
            vals.append(aux)
    combs = product(*vals)
    map_arr = -1*np.ones(N_t)
    i = 0
    for c in combs:
        logi = np.ones(N_t).astype(bool)
        for j in range(n_dim):
            logi = np.logical_and(logi, multi[:, j] == c[j])
        map_arr[logi] = i
        i += 1
    map_arr = map_arr.astype(int)
    return map_arr
