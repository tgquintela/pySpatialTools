
"""
Descriptors
-----------
Collection of precoded descriptors.

"""

from descriptormodel import DescriptorModel
from descriptormodel import DummyDescriptor, NullPhantomDescriptor
from descriptormodel import Interpolator, GeneralDescriptor

from avg_descriptors import AvgDescriptor
from count_descriptor import CountDescriptor, CounterNNDesc
from sum_descriptor import SumDescriptor
from pjensen import PjensenDescriptor
from nbinshistogram import NBinsHistogramDesc, HistogramDistDescriptor
from sparse_counter import SparseCounter
from distances_descriptors import DistancesDescriptor,\
    NormalizedDistanceDescriptor
