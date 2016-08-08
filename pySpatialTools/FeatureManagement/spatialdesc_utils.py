
"""
SpatialDescriptormodel utils
----------------------------
Group different commmon utils to improve the usability for common tasks.

"""

import numpy as np
from pySpatialTools.Retrieve import CircRetriever, KRetriever
from pySpatialTools.FeatureManagement.Descriptors import Interpolator,\
    DummyDescriptor, NullPhantomDescriptor, CounterNNDesc,\
    HistogramDistDescriptor, CountDescriptor
from pySpatialTools.FeatureManagement.features_objects import\
    ImplicitFeatures, PhantomFeatures
from pySpatialTools.FeatureManagement.features_retriever import\
    FeaturesManager
from pySpatialTools.FeatureManagement.spatial_descriptormodels import\
    SpatialDescriptorModel


def create_null_selfindices_desc(locs):
    """Create the PhantomFeatures to be used for the indices."""
    nulldesc = NullPhantomDescriptor()
    feats_ph = PhantomFeatures((len(locs), 1), descriptormodel=nulldesc)
    return feats_ph


def create_pst_interpolation(locs, values, locs_int, Retriever_o,
                             pars_ret, pars_int):
    ## Retriever and main descriptormodel
    pars_ret['ifdistance'] = True
    pars_ret['bool_input_idx'] = True
    ret = Retriever_o(locs, autolocs=locs_int, **pars_ret)
    inter = Interpolator(**pars_int)
    nulldesc = NullPhantomDescriptor()
    vals = np.array(range(len(locs_int)))
    ## Features Retriever creation
    feats = ImplicitFeatures(values, descriptormodel=inter)
    feats_ph = PhantomFeatures((len(locs_int), 1), descriptormodel=nulldesc)
    feats_ret = FeaturesManager([feats, feats_ph],
                                selectors=[(0, 1), (0, 0), (1, 0)],
                                maps_vals_i=vals)
    interpolator = SpatialDescriptorModel(ret, feats_ret)
    return interpolator


def create_pst_histdist_KNN(locs, ks, pars_desc):
    ## Retriever
    pars_ret = {'ifdistance': True, 'bool_input_idx': True}
    pars_desc['ks'] = ks
    pars_ret['info_ret'] = np.max(ks)
    ret = KRetriever(locs, **pars_ret)
    ## Descriptormodel
    vals = np.zeros(len(locs))
    histdistdesc = HistogramDistDescriptor(**pars_desc)
    feats = PhantomFeatures((len(locs), 1), descriptormodel=histdistdesc)
    feats_ph = create_null_selfindices_desc(locs)
    feats_ret = FeaturesManager([feats, feats_ph],
                                selectors=[(0, 1), (0, 0), (1, 0)],
                                maps_vals_i=vals)
    histogramer = SpatialDescriptorModel(ret, feats_ret)
    return histogramer


def create_pst_counter_RNN(locs, radius):
    """Counts the retrieved neighbours by using CircRetriever."""
    pars_ret = {'ifdistance': False, 'bool_input_idx': True}
    pars_ret['info_ret'] = radius
    ret = CircRetriever(locs, **pars_ret)
    ## Features definition
    countNN = CounterNNDesc()
    vals = np.array(range(len(locs)))
    feats = PhantomFeatures((len(locs), 1), descriptormodel=countNN)
    feats_ph = create_null_selfindices_desc(locs)
    feats_ret = FeaturesManager([feats, feats_ph],
                                selectors=[(0, 1), (0, 0), (1, 0)],
                                maps_vals_i=vals)
    counter = SpatialDescriptorModel(ret, feats_ret)
    return counter


def create_counter_types_matrix(locs, types, retriever_o, pars_ret):
    ## 0. Prepare Retriever
    ret = retriever_o(locs, **pars_ret)
    counterdesc = CountDescriptor()
    nulldesc = NullPhantomDescriptor()
    ## 1. Prepare variables
    types_u = np.unique(types)
    types_tr = np.array([np.where(e == types_u)[0][0] for e in types])
    ## 2. Prepare features
    feats = ImplicitFeatures(types_tr, descriptormodel=counterdesc)
    feats_ph = PhantomFeatures((len(locs), 1), descriptormodel=nulldesc)
    ## 3. Prepare builders
    vals = np.arange(len(types))
    feats_ret = FeaturesManager([feats, feats_ph],
                                selectors=[(0, 1), (0, 0), (1, 0)],
                                maps_vals_i=vals)
    counter = SpatialDescriptorModel(ret, feats_ret)
    ## 4. Compute counter
    counts = counter.compute()
    return counts, types


def create_complete_pst_spmodel():
    pass
