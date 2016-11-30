
"""
Example of image feature extraction using pySpatialTools
--------------------------------------------------------
In this example we are gonna use pySpatialTools to extract some features
from an image example. It is not the most recommendable way of do it,
because they are better software to do that specific task, quicker and more
memory efficient.

"""

import mahotas as mh
from pySpatialTools.Retrieve import RetrieverManager, WindowsRetriever
from pySpatialTools.FeatureManagement.features_retriever import\
    FeaturesManager
from pySpatialTools.FeatureManagement.features_objects import\
    ImplicitFeatures
from pySpatialTools.FeatureManagement.Descriptors import NBinsHistogramDesc
from pySpatialTools.FeatureManagement import SpatialDescriptorModel
from pySpatialTools.utils.artificial_data import create_random_image


if __name__ == "__main__":
    #im_example = mh.imread('example.jpg')
    im_example = mh.imread('examples/regular_data/example.jpg')[:, :, [0]]
    im_example = create_random_image((20, 20))[:, :, 0]
    shape = im_example.shape
    feats = im_example.ravel()
    #locs, feats = create_locs_features_from_image(im_example)
    pars_ret, nbins = {'l': 8, 'center': 0, 'excluded': False}, 5
    #windret = WindowsRetriever((10, 10), pars_ret)
    windret = WindowsRetriever(shape, pars_ret)
    binsdesc = NBinsHistogramDesc(nbins)
    features = binsdesc.set_global_info(feats, transform=True)

    gret = RetrieverManager(windret)
    features = ImplicitFeatures(features, descriptormodel=binsdesc,
                                out_type='ndarray')
    feats_ret = FeaturesManager(features, maps_vals_i=('matrix', 400, 400))
    spdesc = SpatialDescriptorModel(gret, feats_ret)
    net = spdesc.compute()
