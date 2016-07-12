
"""
base
----
Module which groups the main *hot spots* of this framework in order to be
called easier.
"""

#import Retrieve.retrievers.Retriever as BaseRetriever
from FeatureManagement.features_objects import Features as BaseFeatures
from FeatureManagement.Descriptors import DescriptorModel as\
    BaseDescriptorModel
from utils.perturbations import GeneralPerturbation as BasePerturbations
from SpatialRelations.relative_positioner import RelativePositioner as\
    BaseRelativePositioner
