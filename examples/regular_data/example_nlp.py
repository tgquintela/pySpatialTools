
"""
Example nlp
-----------
Text is a regular sequential data which could be treated as a
with regular time sampling. Natural language processing uses that view to
process and extract information from text, in the point of view of features
extraction or for measure creation.
"""

import numpy as np
from sklearn.preprocessing import LabelEncoder

from pySpatialTools.Retrieve import RetrieverManager, WindowsRetriever
from pySpatialTools.FeatureManagement.features_retriever import\
    FeaturesManager
from pySpatialTools.FeatureManagement.features_objects import\
    ImplicitFeatures
from pySpatialTools.FeatureManagement.Descriptors import SparseCounter
from pySpatialTools.FeatureManagement import SpatialDescriptorModel


if __name__ == "__main__":
    import time
    t0 = time.time()
    ## Text sample to play with
    with open("examples/regular_data/molly_bloom_joyce.txt") as f:
        words = []
        for line in f:
            words_l = line.replace(',', ' ').replace('\n', ' ')
            words_l = words_l.strip().split(' ')
            words += [w for w in words_l if w != '']

    ## Encoding words into integer labels
    label_encoder = LabelEncoder()
    words_enc = np.array(label_encoder.fit_transform(words))

    ## Computing ngram matrix
    pars_ret, nbins = {'l': 8, 'center': 0, 'excluded': False}, 5
    windret = WindowsRetriever((len(words),), pars_ret)
    gret = RetrieverManager(windret)
    countdesc = SparseCounter()
    features = ImplicitFeatures(words_enc, descriptormodel=countdesc,
                                out_type='dict')
    feats_ret = FeaturesManager(features, maps_vals_i=words_enc)
#    feats_ret = FeaturesManager(words_enc, descriptormodels=countdesc,
#                                maps_vals_i=words_enc)
                                #out='dict', maps_vals_i=words_enc)
    spdesc = SpatialDescriptorModel(gret, feats_ret)
    net = spdesc.compute()
    print time.time()-t0
