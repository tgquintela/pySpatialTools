
"""
Descriptors
-----------
Collection of precoded descriptors.

"""

from descriptormodel import DescriptorModel, DummyDescriptor, GeneralDescriptor
from descriptormodel import Interpolator

from avg_descriptors import AvgDescriptor
from count_descriptor import CountDescriptor
from sum_descriptor import SumDescriptor
from pjensen import PjensenDescriptor
from nbinshistogram import NBinsHistogramDesc
from sparse_counter import SparseCounter
from distances_descriptors import DistancesDescriptor,\
    NormalizedDistanceDescriptor
