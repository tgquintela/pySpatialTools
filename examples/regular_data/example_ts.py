
"""
Example time_series
-------------------
Time series with regular time sampling.
"""

import numpy as np

from pySpatialTools.Retrieve import RetrieverManager, WindowsRetriever
from pySpatialTools.FeatureManagement.features_retriever import\
    FeaturesManager
from pySpatialTools.FeatureManagement.features_objects import ImplicitFeatures
from pySpatialTools.FeatureManagement.Descriptors import NBinsHistogramDesc
from pySpatialTools.FeatureManagement import SpatialDescriptorModel


if __name__ == "__main__":
    import time
    t0 = time.time()
    ## Artificial 1dim random time series
    shape = (2000, )
    ts = np.random.random(shape)

    ## Computing measure by binning
    pars_ret, nbins = {'l': 8, 'center': 0, 'excluded': False}, 5
    windret = WindowsRetriever(shape, pars_ret)
    binsdesc = NBinsHistogramDesc(nbins)
    cat_ts = binsdesc.set_global_info(ts, transform=True)

    gret = RetrieverManager(windret)
    feats_ret = FeaturesManager(ImplicitFeatures(cat_ts, out_type='ndarray',
                                                 descriptormodel=binsdesc),
                                maps_vals_i=cat_ts)

#    feats_ret = FeaturesManager(cat_ts, descriptormodels=binsdesc,
#                                maps_vals_i=cat_ts)

    spdesc = SpatialDescriptorModel(gret, feats_ret)
    net = spdesc.compute()

    # Compare with the expected result
    try:
        np.testing.assert_allclose(net.sum(1)/net.sum(), 1./nbins, atol=0.03)
    except AssertionError as e:
        print(e)
    print time.time()-t0
