
"""
test_spdescriptormodels
-----------------------
testing spatial descriptor models utilities.

"""


import numpy as np
import os
import copy

## Retrieve
from pySpatialTools.Discretization import GridSpatialDisc
from pySpatialTools.Retrieve import create_retriever_input_output,\
    SameEleNeigh, KRetriever, CircRetriever, RetrieverManager
# Artificial data
from pySpatialTools.utils.artificial_data import generate_randint_relations

## Features
from pySpatialTools.FeatureManagement.features_retriever import\
    FeaturesManager
from pySpatialTools.FeatureManagement.features_objects import Features,\
    ImplicitFeatures, ExplicitFeatures

from pySpatialTools.utils.perturbations import PermutationPerturbation
from pySpatialTools.utils.util_classes import create_mapper_vals_i

## Descriptormodel
from pySpatialTools.FeatureManagement.Descriptors import Countdescriptor,\
    AvgDescriptor
from pySpatialTools.FeatureManagement import SpatialDescriptorModel

from ..utils.util_external.Logger import Logger


def test():
    n, nx, ny = 1000, 100, 100
    locs = np.random.random((n, 2))*10
    ## Retrievers management
    ret0 = KRetriever(locs, 3, ifdistance=True)
    ret1 = CircRetriever(locs, .3, ifdistance=True)
    #countdesc = Countdescriptor()

    # Creation of retriever of regions
    griddisc = GridSpatialDisc((nx, ny), (0, 10), (0, 10))
    regdists = generate_randint_relations(0.001, (nx, ny), p0=0., maxvalue=1)
    regret = SameEleNeigh(regdists)
    m_in, m_out = create_retriever_input_output(griddisc.discretize(locs))
    regret._output_map = [m_out]
    gret = RetrieverManager([ret0, ret1, regret])

    ## Features management
    feat_arr0 = np.random.randint(0, 20, (n, 1))

    features = ImplicitFeatures(feat_arr0)
    reindices = np.vstack([np.random.permutation(n) for i in range(5)])
    perturbation = PermutationPerturbation(reindices.T)

    features.add_perturbations(perturbation)

    def map_vals_i_t(s, i, k):
        k_p, k_i = s.features[0]._map_perturb(k)
        i_n = s.features[0]._perturbators[k_p].apply2indice(i, k_i)
        return feat_arr0[i_n].ravel()[0]
    map_vals_i = create_mapper_vals_i(map_vals_i_t, feat_arr0)

    countdesc = Countdescriptor()
    feats_ret = FeaturesManager([features], countdesc, maps_vals_i=map_vals_i)
    feats_ret.add_aggregations((locs, griddisc), regret)

    ## Descriptor
    #avgdesc = AvgDescriptor(feats_ret)
    spdesc = SpatialDescriptorModel(gret, feats_ret)
    nets = spdesc.compute()
    spdescs = []
    idxs = [slice(0, 250, 1), slice(250, 500, 1), slice(500, 750, 1)]
    idxs += [slice(750, 1000, 1)]
    for i in range(4):
        aux_spdesc = copy.copy(spdesc)
        aux_spdesc.set_loop(idxs[i])
        spdescs.append(aux_spdesc)
    netss = [spdescs[i].compute() for i in range(4)]

    try:
        logfile = Logger('logfile.log')
        spdesc.compute_process(logfile, lim_rows=0, n_procs=0)
        os.remove('logfile.log')
    except:
        raise Exception("Not usable compute_process.")
