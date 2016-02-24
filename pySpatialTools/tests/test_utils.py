
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
from pySpatialTools.utils.artificial_data import randint_sparse_matrix
from ..utils.util_classes import Locations, SpatialElementsCollection,\
    Membership


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
    locs._check_coord(0)
    locs._check_coord(locs1[0])
    locs._check_coord([0, 3])
    locs._check_coord([locs1[0], locs1[3]])
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

    ## Membership
    n_in, n_out = 100, 20
    relations = [np.unique(np.random.randint(0, n_out,
                                             np.random.randint(n_out)))
                 for i in range(n_in)]
    relations = [list(e) for e in relations]
    memb1 = Membership(relations)

    memb1.to_network()
    memb1.to_dict()
    memb1.to_sparse()
    memb1.reverse_mapping()
    memb1.getcollection(0)
    memb1.collections_id
    memb1.n_collections
    memb1.n_elements
    memb1.membership
    str(memb1)
    memb1[0]
    memb1 == 0
    for e in memb1:
        pass

    memb2 = Membership(np.random.randint(0, 20, 100))
    memb2.to_network()
    memb2.to_dict()
    memb2.to_sparse()
    memb2.reverse_mapping()
    memb2.getcollection(0)
    memb2.collections_id
    memb2.n_collections
    memb2.n_elements
    memb2.membership
    str(memb2)
    memb2[0]
    memb2 == 0
    for e in memb2:
        pass

    sparse = randint_sparse_matrix(0.2, (200, 100), 1)
    memb3 = Membership(sparse)
    memb3.to_network()
    memb3.to_dict()
    memb3.to_sparse()
    memb3.reverse_mapping()
    memb3.getcollection(0)
    memb3.collections_id
    memb3.n_collections
    memb3.n_elements
    memb3.membership
    str(memb3)
    memb3[0]
    memb3 == 0
    for e in memb3:
        pass

    relations = [[np.random.randint(10)] for i in range(50)]
    memb4 = Membership(relations)
    memb4.to_network()
    memb4.to_dict()
    memb4.to_sparse()
    memb4.reverse_mapping()
    memb4.getcollection(0)
    memb4.collections_id
    memb4.n_collections
    memb4.n_elements
    memb4.membership
    str(memb4)
    memb4[0]
    memb4 == 0
    for e in memb4:
        pass

    relations[0].append(0)
    memb5 = Membership(relations)
    memb5.to_network()
    memb5.to_dict()
    memb5.to_sparse()
    memb5.reverse_mapping()
    memb5.getcollection(0)
    memb5.collections_id
    memb5.n_collections
    memb5.n_elements
    memb5.membership
    str(memb5)
    memb5[0]
    memb5 == 0
    for e in memb5:
        pass

    relations[0].append(0)
    memb6 = Membership((sparse, np.arange(100)))
    memb6.to_network()
    memb6.to_dict()
    memb6.to_sparse()
    memb6.reverse_mapping()
    memb6.getcollection(0)
    memb6.collections_id
    memb6.n_collections
    memb6.n_elements
    memb6.membership
    str(memb6)
    memb6[0]
    memb6 == 0
    for e in memb6:
        pass
