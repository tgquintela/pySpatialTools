

from aggregation_utils import compute_aggregate_counts,\
    average_position_by_aggarr
#from Mscthesis.IO.io_aggfile import read_aggregation


class Aggregator:
    "Aggregate or read aggregate information."

    def __init__(self, filepath=None, typevars=None):
        """It is needed the identification of the aggvar and feat_vars with the
        filepath if we want to get a aggregation from a file.
        """
        self.typevars = format_typevars(typevars)
        if filepath is None:
            self.bool_read_agg = False
        else:
            self.bool_read_agg = True
            self.filepath = filepath

    def retrieve_aggregation(self, df=None, reindices=None, funct=None):
        "Main function for retrieving aggregation."
        if self.bool_read_agg:
            # TODO: Function to read file
            filepath, typevars = self.filepath, self.typevars
            agglocs, aggfeatures = read_aggregation(filepath, typevars)
        else:
            ## Correct inputs
            #################
            locs = df[self.typevars['loc_vars']].as_matrix()
            feat_arr = df[self.typevars['feat_vars']].as_matrix()
            agg_arr = df[self.typevars['agg_var']].as_matrix()
#            if self.typevars['agg_var'] is None:
#                ## Check if discretized or to do
#                if type(self.discretizor) == np.ndarray:
#                    agg_arr = self.discretizor
#                    agglocs = average_position_by_aggarr(locs, agg_arr)
#                else:
#                    agg_arr = self.discretizor.map2id(locs)
#                    agglocs = self.discretizor.discretize(locs)
#                self.typevars['agg_var'] = 'aggvar'
#            else:
#                agg_arr = df[self.typevars['agg_var']].as_matrix()
#                agglocs = average_position_by_aggarr(locs, agg_arr)
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

#    def retrieve_aggloc(self, point_i):
#        point_i = point_i.reshape(1, 2)
#        region = self.spatial_disc.map2id(point_i)
#        return region
#
#    def retrieve_agglocs(self, locs):
#        regions = self.spatial_disc.map2id(locs)
#        return regions


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
