
"""
tests
=====
Module which contains the functions required to test how good performs a
prediction out-of-sample.

"""

import time
import numpy as np
import warnings

###############################################################################
############################## Testing function ###############################
###############################################################################
import test_utils
import test_spatial_discretizer
import test_retriever
import test_spatial_relations
import test_features_retriever
import test_perturbation
import test_descriptormodels
import test_spdescriptormodels

# Messages
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


# Test
def test():
    "Please, try to update that everytime it is a change in the tests."
    ## Init tests
    t0 = time.time()
    print("")
    print(message_init_tests)
    print(ref_computer_stats)
    print(message_ref_computer % time_ref_computer)
    ## Performing tests
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        test_utils.test()
        test_spatial_discretizer.test()
        test_retriever.test()
        test_spatial_relations.test()
        test_features_retriever.test()
        test_perturbation.test()
        test_descriptormodels.test()
        test_spdescriptormodels.test()
    ## Closing tests
    time_own_computer = str(np.round(time.time()-t0, 2))
    print(message_own_computer % time_own_computer)
