
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

from implicit_retrievers import KRetriever, CircRetriever, WindowsRetriever
from explicit_retrievers import SameEleNeigh, OrderEleNeigh,\
    LimDistanceEleNeigh

'''
retrievers = {'sameeleneigh': SameEleNeigh, 'ordereleneigh': OrderEleNeigh,
              'limdistanceneigh': LimDistanceEleNeigh,
              'kretriever': KRetriever, 'circretriever': CircRetriever,
              'windowsretriever': WindowsRetriever}
features = {}

    feats1 = ImplicitFeatures(featsarr0)

    m_vals_i = np.random.randint(0, 5, 50)
    ret = CircRetriever(locs1, autolocs=locs_input, info_ret=3,
                        bool_input_idx=True)
    feat = FeaturesManager(feats1, maps_vals_i=m_vals_i, mode='sequential',
                           descriptormodels=None)
    spdesc = SpatialDescriptorModel(retrievers=ret, featurers=feat,
                                    mapselector_spdescriptor=None,
                                    perturbations=perturbation,
                                    aggregations=None, name_desc=n_desc)
    ## Complete processes
    spdesc.compute()
'''

def create_null_selfindices_desc(locs):
    """Create the PhantomFeatures to be used for the indices.

    Parameters
    ----------
    locs: np.ndarray, shape (n, n_dim) or list
        the point locations considered

    Returns
    -------
    feats_ph: pst.PhantomFeatures
        the features.

    """
    nulldesc = NullPhantomDescriptor()
    feats_ph = PhantomFeatures((len(locs), 1), descriptormodel=nulldesc)
    return feats_ph


def create_pst_interpolation(locs, values, locs_int, Retriever_o,
                             pars_ret, pars_int):
    """Create interpolation spatial descriptor model.

    Parameters
    ----------
    locs: array_like or list
        the spatial information
    values: array_like or list
        the values of each of the retrievable elements.
    locs_int: array_like or list
        the spatial information of the elements to obtain values by
        interpolation.
    Retriever_o: pst.BaseRetriever
        the retriever class (not instantiated yet).
    pars_ret: dict
        the parameters of the retriever.
    pars_int: dict
        the parameters of the interpolation definition.

    Returns
    -------
    interpolator: pst.SpatialDescriptorModel
        the spatial descriptor model for interpolation.

    """
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
    """Creation of the model of KNN histogram.

    Parameters
    ----------
    locs: array_like, shape (n, n_dim) or list
        the spatial information of the retrievable elements.
    ks: int, list or array_like
        number of perturbation to considered.
    pars_desc: dict
        the parameters of the descriptormodel.

    Returns
    -------
    histogramer: pst.SpatialDescriptormodel
        the spatial descriptormodel based on computing the histogram.

    """
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
    """Counts the retrieved neighbours by using CircRetriever.

    Parameters
    ----------
    locs: array_like, shape (n, n_dim) or list
        the spatial information of the retrievable elements.
    radius: float or array_like, shape (n)
        the radius of the retrievable neighbourhood.

    Returns
    -------
    counter: pst.SpatialDescriptorModel
        the counter spatial descriptor model based on count neighbours.

    """
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
    """Create network of types.

    Parameters
    ----------
    locs: array_like, shape (n, n_dim)
        the spatial information of the retrievable elements.
    types: array_like or list
        the types codes.
    retriever_o: pst.BaseRetriever
        the retriever class (not instantiated yet).
    pars_ret: dict
        the parameters of the retriever.

    Returns
    -------
    counts: np.ndarray or scipy.sparse
        the counts matrix.
    types_u: array_like or list
        the types codes in the same order of the counts matrix.
    """
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
    return counts, types_u


def create_complete_pst_spmodel():
    pass
