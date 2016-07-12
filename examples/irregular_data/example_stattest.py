
"""
Example stattest
----------------
Spatial statistical testing examples.

"""

import numpy as np
from pySpatialTools.Retrieve import KRetriever, CircRetriever
from pySpatialTools.utils.perturbations import JitterLocations,\
    PermutationPerturbation
from pySpatialTools.FeatureManagement.Descriptors import CountDescriptor
from pySpatialTools.FeatureManagement.features_objects import ImplicitFeatures
from pySpatialTools.FeatureManagement.features_retriever import\
    FeaturesManager
from pySpatialTools.FeatureManagement import SpatialDescriptorModel


if __name__ == "__main__":
    ## Parameters initialization
    nx, ny = 3, 3
    nclass = 20
    nneighs = 3
    n_points, classes = [], []
#    for i in range(nx):
#        for j in range(ny):
#            points.append(np.vstack([np.random.normal(i, 0.2, size=nclass),
#                                     np.random.normal(j, 0.2, size=nclass)]))
#            classes.append(np.ones(nclass)*(ny*i+j))
#    points = np.hstack(points).T
#    classes = np.hstack(classes)

    ## 1-element creation
    points = np.random.random((100, 2))
    for i in range(len(points)):
        for k in range(nneighs):
            aux_x = points[i, 0] + np.random.normal(0, 0.01, size=nclass)
            aux_y = points[i, 1] + np.random.normal(0, 0.01, size=nclass)
            aux = np.vstack([aux_x, aux_y])
            n_points.append(aux)
            classes.append(np.arange(nclass))
    n_points = np.hstack(n_points).T
    classes = np.hstack(classes)

    # Retrievers
    perturbations = JitterLocations(0.001, 250)
    ret0 = KRetriever(locs=n_points, info_ret=nclass*nneighs,
                      autolocs=points, ifdistance=False,
                      perturbations=perturbations)
    ret1 = CircRetriever(locs=n_points, info_ret=0.025,
                         autolocs=points, ifdistance=False,
                         perturbations=perturbations)
    ############### WARNING: TOTEST (not shape)
    # Features and descriptor
    desc = CountDescriptor()
    names = [str(e) for e in range(nclass)]
    feats = ImplicitFeatures(classes, descriptormodel=desc,  # names=names,
                             perturbations=perturbations)
    feats_ret = FeaturesManager(feats, maps_vals_i=np.zeros((len(points), 1)))
    measurer = SpatialDescriptorModel(ret0, feats_ret)
    measure = measurer.compute()

    ###########################################################################
    ######### Statistical tests
    ## Detect which one is the more stable under jitter and permutation
    new_points = np.vstack([n_points, points])
    new_classes = np.hstack([classes, np.ones(len(points))*nclass])

    # Perturbations
    jitter_perturbs = JitterLocations(0.001, 250)
    perm_perturbs = PermutationPerturbation((len(new_points), 250))

    ## 1st test (jitter)  [TODO: error heterogeneous shape retrievers]
    # a) using K-neighs neighborhood
    ret0 = KRetriever(locs=new_points[:], info_ret=nclass*nneighs,
                      ifdistance=False)
    feats = ImplicitFeatures(new_classes[:], descriptormodel=desc)
    feats_ret = FeaturesManager(feats, maps_vals_i=np.zeros((len(new_points))))
    measurer = SpatialDescriptorModel(ret0, feats_ret,
                                      perturbations=jitter_perturbs)
    measure = measurer.compute()
    # b) using Radius neighborhood
    ret1 = CircRetriever(locs=new_points[:], info_ret=0.025, ifdistance=False)
    feats = ImplicitFeatures(new_classes[:], descriptormodel=desc)
    feats_ret = FeaturesManager(feats, maps_vals_i=np.zeros((len(new_points))))
    measurer = SpatialDescriptorModel(ret1, feats_ret,
                                      perturbations=jitter_perturbs)
    measure = measurer.compute()

    ## 2nd test (permutation)
    # a) using K-neighs neighborhood
    ret0 = KRetriever(locs=new_points[:], info_ret=nclass*nneighs,
                      ifdistance=False)
    feats = ImplicitFeatures(new_classes[:], descriptormodel=desc)
    feats_ret = FeaturesManager(feats, maps_vals_i=np.zeros((len(new_points))))
    measurer = SpatialDescriptorModel(ret0, feats_ret,
                                      perturbations=perm_perturbs)
    measure = measurer.compute()
    # b) using Radius neighborhood
    ret1 = CircRetriever(locs=new_points[:], info_ret=0.025, ifdistance=False)
    feats = ImplicitFeatures(new_classes[:], descriptormodel=desc)
    feats_ret = FeaturesManager(feats, maps_vals_i=np.zeros((len(new_points))))
    measurer = SpatialDescriptorModel(ret1, feats_ret,
                                      perturbations=perm_perturbs)
    measure = measurer.compute()

#    ## 3rd test (random sampling)
#    # a) using K-neighs neighborhood
#    ret0 = KRetriever(locs=new_points, info_ret=nclass*nneighs,
#                      ifdistance=False)
#    measurer = SpatialDescriptorModel(ret0, feats,
#                                      perturbations=jitter_perturbs)
#    measure = measurer.compute()
#    # b) using Radius neighborhood
#    ret1 = CircRetriever(locs=new_points, info_ret=0.025, ifdistance=False)
#    measurer = SpatialDescriptorModel(ret0, feats,
#                                      perturbations=jitter_perturbs)
#    measure = measurer.compute()
