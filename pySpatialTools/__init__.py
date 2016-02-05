
from version import *

'''
import Correlation_Models
import Descriptor_Models
import Feature_engineering
import Geo_tools
import Interpolation
import IO
import Preprocess
import Recommender
import Retrieve
import Sampling
import Simulations
import Spatial_Relations
import Tests
import utils
'''


###############################################################################
############################## Testing function ###############################
###############################################################################
# TODO: test_spatial_relations
from Tests import test_spatial_discretizer, test_retriever,\
    test_features_retriever, test_perturbation, test_descriptormodels,\
    test_spdescriptormodels
from Tests import test_spatial_relations
import time
import numpy as np

message_init_tests =\
    "***---*** Testing python package pySpatialTools ***---***\n%s" % ('-'*57)
ref_computer_stats = """Test compared with a reference computer with specs:
Linux Mint 17 Qiana
Kernel Linux 3.13.0-24-generic
Intel Core i7-3537U
4GB de RAM
NVidia GeForce GT720M 2GB
-------------------------------"""
message_ref_computer = "Average time in ref. computer: %s seconds."
time_ref_computer = str(20.58)
message_own_computer = "Time testing in this computer: %s seconds."


def test():
    "Please, try to update that everytime it is a change in the tests."
    ## Init tests
    t0 = time.time()
    print("\n")
    print(message_init_tests)
    print(ref_computer_stats)
    ## Performing tests
    test_spatial_discretizer.test()
    test_retriever.test()
    test_features_retriever.test()
    test_perturbation.test()
    test_descriptormodels.test()
    test_spdescriptormodels.test()
    ## Closing tests
    time_own_computer = str(np.round(time.time()-t0, 2))
    print(message_ref_computer % time_ref_computer)
    print(message_own_computer % time_own_computer)
