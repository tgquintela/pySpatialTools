
"""
test utilities
--------------
testing functions which acts as a utils.

"""

m = """In fiction, artificial intelligence is often seen a menace that allows
robots to take over and enslave humanity. While some share these concerns in
real life, a researcher suggests the robot-conquest might be more subtle than
imagined.
According to Moshe Vardi, director of the Institute for Information Technology
at Rice University in Texas, in the coming 30 years, advanced robots will
threaten tens of millions of jobs.
"""

import numpy as np
from ..utils.util_classes import Locations, SpatialElementsCollection


def test():
    words = m.replace('\n', ' ').replace('.', ' ').strip().split(" ")
    ids = [hash(e) for e in words]
    functs = [lambda x: str(x)+words[i] for i in range(len(words))]
    locs1 = np.random.random((100, 5))
    locs2 = np.random.random((100, 1))
    locs3 = np.random.random(100)
    locs4 = np.random.random((100, 2))

    ## Testing locations
    #SpatialElementsCollection(words)
    #SpatialElementsCollection(ids)
    #SpatialElementsCollection(functs)
    sptrans = lambda x, p: np.sin(x)

    locs = Locations(locs1)
    assert((locs == locs1[0])[0])
    locs.compute_distance(locs[1])
    locs.space_transformation(sptrans, {})
    locs.in_radio(locs[0], 0.2)

    locs = Locations(locs2)
    assert((locs == locs2[0])[0])
    locs.compute_distance(locs[1])
    locs.space_transformation(sptrans, {})
    locs.in_manhattan_d(locs[0], 0.2)

    locs = Locations(locs3)
    assert((locs == locs3[0])[0])
    locs.compute_distance(locs[1])
    locs.space_transformation(sptrans, {})

    locs = Locations(locs4)
    locs.in_block_distance_d(np.random.random((1, 2)), 0.2)
