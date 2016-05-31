
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
from pySpatialTools.utils.artificial_data import randint_sparse_matrix,\
    generate_randint_relations, generate_random_relations_cutoffs,\
    random_transformed_space_points, random_space_points, create_random_image,\
    random_shapely_polygon, random_shapely_polygons,\
    continuous_array_features, categorical_agg_dict_features,\
    categorical_array_features, continuous_dict_features,\
    categorical_dict_features, continuous_agg_array_features,\
    categorical_agg_array_features, continuous_agg_dict_features
from pySpatialTools.utils.artificial_data.artificial_measure import *

from pySpatialTools.utils.artificial_data.artificial_data_membership import\
    random_membership
from pySpatialTools.utils.mapper_vals_i import create_mapper_vals_i,\
    Map_Vals_i
from ..utils.util_classes import Locations, SpatialElementsCollection,\
    Membership
from pySpatialTools.utils.selectors import DummySelector,\
    GeneralCollectionSelectors, Spatial_RetrieverSelector,\
    Feat_RetrieverSelector, FeatInd_RetrieverSelector, Desc_RetrieverSelector,\
    Sp_DescriptorSelector


def test():
    ###########################################################################
    ############################# Artificial data #############################
    ###########################################################################
    ## Random relations
    n, density, shape = 100, 0.1, (10, 10)
    randint_sparse_matrix(density, shape, maxvalue=10)
    generate_randint_relations(density, shape, p0=0., maxvalue=1)
    generate_random_relations_cutoffs(n, 0.5, 0.9, True, 'network')
    generate_random_relations_cutoffs(n, 0.5, 0.9, False, 'network')
    generate_random_relations_cutoffs(n, 0.5, 0.9, True, 'sparse')

    n_elements, n_collections = 100, 10
    random_membership(n_elements, n_collections, multiple=True)
    random_membership(n_elements, n_collections, multiple=False)

    ## Random points
    n_points, n_dim, funct = 100, 2, np.cos
    random_transformed_space_points(n_points, n_dim, funct)
    random_transformed_space_points(n_points, n_dim, None)
    random_space_points(n_points, n_dim)

    ## Artificial grid data
    create_random_image(shape, n_modes=1)
    create_random_image(shape, n_modes=3)

    ## Artificial regions
    n_poly = 10
    random_shapely_polygon(bounding=(None, None), n_edges=0)
    random_shapely_polygon(bounding=((0., 1.), None), n_edges=0)
    random_shapely_polygon(bounding=(None, None), n_edges=4)
    random_shapely_polygons(n_poly, bounding=(None, None), n_edges=0)

    ## Artificial random features
    n, n_feats = np.random.randint(10, 1000), np.random.randint(2, 20)
    n_feats2 = [np.random.randint(2, 20) for i in range(n_feats)]
    ks = np.random.randint(1, 20)

    feats = continuous_array_features(n, n_feats)
    assert(len(feats.shape) == 2)
    feats = categorical_array_features(n, n_feats)
    assert(len(feats.shape) == 2)
    feats = categorical_array_features(n, n_feats2)
    assert(len(feats.shape) == 2)
    feats = continuous_dict_features(n, n_feats)
    assert(type(feats[0]) == dict)
    feats = categorical_dict_features(n, n_feats)
    assert(type(feats[0]) == dict)

    feats = continuous_agg_array_features(n, n_feats, ks)
    assert(len(feats.shape) == 3)
    feats = categorical_agg_array_features(n, n_feats, ks)
    assert(len(feats.shape) == 3)
    feats = categorical_agg_array_features(n, n_feats2, ks)
    assert(len(feats.shape) == 3)
    feats = continuous_agg_dict_features(n, n_feats, ks)
    assert(type(feats[0][0]) == dict)
    feats = categorical_agg_dict_features(n, n_feats, ks)
    assert(type(feats[0][0]) == dict)

    ## Artificial measures
    n_vals_i, n_iss = np.random.randint(2, 30), np.random.randint(1, 30)

    create_empty_features_array(n_feats, n_iss, ks)
    create_empty_features_dict(n_feats, n_iss, ks)
    create_features_i_array(n_feats, n_iss, ks)
    create_features_i_dict(n_feats, n_iss, ks)

    create_vals_i(n_iss, n_vals_i, ks)

    create_empty_array(ks, n_vals_i, n_feats)
    create_empty_append(ks, n_iss, n_feats)
    create_empty_replacelist(ks, n_iss, n_feats)

    create_artificial_measure_array(ks, n_vals_i, n_feats)
    create_artificial_measure_append(ks, n_vals_i, n_feats)
    create_artificial_measure_replacelist(ks, n_vals_i, n_feats)
    create_artificial_measure_replacelist(ks, n_vals_i, n_feats, True)

    ###########################################################################
    ############################ Spatial Elements #############################
    ###########################################################################
    ## Parameters
    words = m.replace('\n', ' ').replace('.', ' ').strip().split(" ")
    ids = [hash(e) for e in words]
    functs = [lambda x: str(x)+words[i] for i in range(len(words))]
    regs = random_shapely_polygons(10, bounding=(None, None), n_edges=0)

    ## Testing Elemets
    words_id = np.arange(len(words))
    words_elements = SpatialElementsCollection(words, words_id)
    words_elements2 = SpatialElementsCollection(words, list(words_id))
    words_elements = SpatialElementsCollection(words)
    ids_elements = SpatialElementsCollection(ids)
    functs_elements = SpatialElementsCollection(functs)
    polys_elements = SpatialElementsCollection(regs, np.arange(len(regs)))

    # Testing error instantiation
    try:
        flag_error = False
        SpatialElementsCollection(0)
        flag_error = True
        raise Exception("It has to halt here.")
    except:
        if flag_error:
            raise Exception("It has to halt here.")
    try:
        flag_error = False
        SpatialElementsCollection(words, np.arange(len(words)+1))
        flag_error = True
        raise Exception("It has to halt here.")
    except:
        if flag_error:
            raise Exception("It has to halt here.")
    try:
        flag_error = False
        tags = range(len(words)) + [len(words)-1]
        SpatialElementsCollection(words, tags)
        flag_error = True
        raise Exception("It has to halt here.")
    except:
        if flag_error:
            raise Exception("It has to halt here.")
    try:
        flag_error = False
        SpatialElementsCollection(words, 5)
        flag_error = True
        raise Exception("It has to halt here.")
    except:
        if flag_error:
            raise Exception("It has to halt here.")

    # Class functions
    words_elements[0]
    try:
        flag_error = False
        words_elements[len(words_elements)]
        flag_error = True
        raise Exception("It has to halt here.")
    except:
        if flag_error:
            raise Exception("It has to halt here.")
    try:
        flag_error = False
        words_elements2[words[0]]
        flag_error = True
        raise Exception("It has to halt here.")
    except:
        if flag_error:
            raise Exception("It has to halt here.")

    words_elements.elements_id = None
    try:
        flag_error = False
        words_elements[words[0]]
        flag_error = True
        raise Exception("It has to halt here.")
    except:
        if flag_error:
            raise Exception("It has to halt here.")

    words_elements[0]

    for e in words_elements:
        pass

    for e in words_elements2:
        pass

    words_elements == words[0]
    relabel_map = np.arange(len(words))
    try:
        flag_error = False
        words_elements.relabel_elements(range(len(words)))
        flag_error = True
    except:
        if flag_error:
            raise Exception("It has to halt here.")

    words_elements.relabel_elements(relabel_map)
    relabel_map = dict(zip(relabel_map, relabel_map))
    words_elements.relabel_elements(relabel_map)

    ids_elements[0]
    for e in ids_elements:
        pass
    ids_elements == words[0]

    functs_elements[0]
    for e in functs_elements:
        pass
    functs_elements == words[0]

    # Polygon collections
    polys_elements == polys_elements[0]

    ############################ Locations Object #############################
    ###########################################################################
    ## Locations
    locs1 = np.random.random((100, 5))
    locs2 = np.random.random((100, 1))
    locs3 = np.random.random(100)
    locs4 = np.random.random((100, 2))
    sptrans = lambda x, p: np.sin(x)

    class Translocs:
        def __init__(self):
            pass

        def apply_transformation(self, x, p={}):
            return sptrans(x, p)
    sptrans2 = Translocs()

    lspcol = SpatialElementsCollection(locs1, np.arange(len(locs1)))
    lspcol == lspcol[0]

    try:
        flag_error = False
        locs = Locations(locs1, 5)
        flag_error = True
        raise Exception("It has to halt here.")
    except:
        if flag_error:
            raise Exception("It has to halt here.")
    try:
        flag_error = False
        locs = Locations(locs1, list(range(len(locs1)+1)))
        flag_error = True
        raise Exception("It has to halt here.")
    except:
        if flag_error:
            raise Exception("It has to halt here.")
    try:
        flag_error = False
        tags = list(range(len(locs1)))
        tags[0] = 1
        locs = Locations(locs1, tags)
        flag_error = True
        raise Exception("It has to halt here.")
    except:
        if flag_error:
            raise Exception("It has to halt here.")

    locs = Locations(locs1)
    locsbis = Locations(locs1, list(range(len(locs1))))
    for l in locs:
        pass
    try:
        flag_error = False
        locs[-1]
        flag_error = True
        raise Exception("It has to halt here.")
    except:
        if flag_error:
            raise Exception("It has to halt here.")
    try:
        flag_error = False
        locsbis[slice(0, 9)]
        flag_error = True
        raise Exception("It has to halt here.")
    except:
        if flag_error:
            raise Exception("It has to halt here.")

    locsbis[0]
    locs[0]
    assert((locs == locs1[0])[0])
    locs.compute_distance(locs[1])
    locs.space_transformation(sptrans, {})
    locs.space_transformation(sptrans2, {})
    locs._check_coord(0)
    locs._check_coord(locs[0])
    locs._check_coord([0, 3])
    locs._check_coord(np.random.random(locs.locations.shape[1]))
    locs._check_coord([locs1[0], locs1[3]])
    locs._check_coord(None)
    locs.in_radio(locs[0], 0.2)
    locs.data

    locs = Locations(locs2)
    assert((locs == locs2[0])[0])
    locs.compute_distance(locs[1])
    locs.space_transformation(sptrans, {})
    locs.space_transformation(sptrans2, {})
    locs.in_manhattan_d(locs[0], 0.2)

    locs = Locations(locs3)
    assert((locs == locs3[0])[0])
    locs.compute_distance(locs[1])
    locs.space_transformation(sptrans, {})
    locs.space_transformation(sptrans2, {})

    locs = Locations(locs4)
    locs.in_block_distance_d(np.random.random((1, 2)), 0.2)

    ###########################################################################
    ############################### Membership ################################
    ###########################################################################
    # artificial data
    random_membership(10, 20, True)
    random_membership(10, 20, False)

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
    memb1.getcollection(memb1.max_collection_id-1)
    memb1.collections_id
    memb1.n_collections
    memb1.n_elements
    memb1.membership
    str(memb1)
    memb1[0]
    memb1 == 0
    for e in memb1:
        pass

#    op2 = np.all([t == dict for t in types])
    relations = [dict(zip(e, len(e)*[{'membership': 1}])) for e in relations]
    memb1_dict = Membership(relations)
    memb1_dict.to_network()
    memb1_dict.to_dict()
    memb1_dict.to_sparse()
    memb1_dict.reverse_mapping()
    memb1_dict.getcollection(0)
    memb1.getcollection(memb1.max_collection_id-1)
    memb1_dict.collections_id
    memb1_dict.n_collections
    memb1_dict.n_elements
    memb1_dict.membership
    memb1.shape
    memb1.max_collection_id

    memb2 = Membership(np.random.randint(0, 20, 100))
    memb2.to_network()
    memb2.to_dict()
    memb2.to_sparse()
    memb2.reverse_mapping()
    memb2.getcollection(0)
    memb2.getcollection(memb2.max_collection_id-1)
    memb2.collections_id
    memb2.n_collections
    memb2.n_elements
    memb2.membership
    str(memb2)
    memb2[0]
    memb2 == 0
    for e in memb2:
        pass
    memb2.shape
    memb2.max_collection_id

    sparse = randint_sparse_matrix(0.2, (200, 100), 1)
    memb3 = Membership(sparse)
    memb3.to_dict()
    memb3.to_network()
    memb3.to_sparse()
    memb3.reverse_mapping()
    memb3.getcollection(0)
    memb3.getcollection(memb3.max_collection_id-1)
    memb3.collections_id
    memb3.n_collections
    memb3.n_elements
    memb3.membership
    str(memb3)
    memb3[0]
    memb3 == 0
    for e in memb3:
        pass
    memb3.shape
    memb3.max_collection_id

    relations = [[np.random.randint(10)] for i in range(50)]
    memb4 = Membership(relations)
    memb4.to_network()
    memb4.to_dict()
    memb4.to_sparse()
    memb4.reverse_mapping()
    memb4.getcollection(0)
    memb4.getcollection(memb4.max_collection_id-1)
    memb4.collections_id
    memb4.n_collections
    memb4.n_elements
    memb4.membership
    str(memb4)
    memb4[0]
    memb4 == 0
    for e in memb4:
        pass
    memb4.shape
    memb4.max_collection_id

    relations[0].append(0)
    memb5 = Membership(relations)
    memb5.to_network()
    memb5.to_dict()
    memb5.to_sparse()
    memb5.reverse_mapping()
    memb5.getcollection(0)
    memb5.getcollection(memb5.max_collection_id-1)
    memb5.collections_id
    memb5.n_collections
    memb5.n_elements
    memb5.membership
    str(memb5)
    memb5[0]
    memb5 == 0
    for e in memb5:
        pass
    memb5.shape
    memb5.max_collection_id

    relations[0].append(0)
    memb6 = Membership((sparse, np.arange(100)))
    memb6.to_network()
    memb6.to_dict()
    memb6.to_sparse()
    memb6.reverse_mapping()
    memb6.getcollection(0)
    memb6.getcollection(memb6.max_collection_id-1)
    memb6.collections_id
    memb6.n_collections
    memb6.n_elements
    memb6.membership
    str(memb6)
    memb6[0]
    memb6 == 0
    for e in memb6:
        pass
    memb6.shape
    memb6.max_collection_id

    ###########################################################################
    ############################### Mapper vals ###############################
    ###########################################################################
    feat_arr0 = np.random.randint(0, 20, 100)

    def map_vals_i_t(s, i, k):
        k_p, k_i = s.features[0]._map_perturb(k)
        i_n = s.features[0]._perturbators[k_p].apply2indice(i, k_i)
        return feat_arr0[i_n].ravel()[0]
    map_vals_i = create_mapper_vals_i(map_vals_i_t, feat_arr0)

    # correlation
    map_vals_i = create_mapper_vals_i('correlation', feat_arr0)
    map_vals_i = create_mapper_vals_i(('correlation', 100, 20), feat_arr0)
    map_vals_i = create_mapper_vals_i('matrix')
    map_vals_i = create_mapper_vals_i('matrix', feat_arr0)
    map_vals_i = create_mapper_vals_i('matrix', feat_arr0.reshape((100, 1)))
    map_vals_i = create_mapper_vals_i(('matrix', 20), list(feat_arr0))
    map_vals_i = create_mapper_vals_i(('matrix', 100, 20), len(feat_arr0))
    map_vals_i = create_mapper_vals_i('matrix', slice(0, 100, 1))
    map_vals_i.set_prefilter(slice(0, 100, 1))
    map_vals_i.set_prefilter(10)
    map_vals_i.set_prefilter([0, 2])
    map_vals_i.set_sptype('correlation')
    map_vals_i[(None, [0], 0)]
    map_vals_i.apply(None, [0], 0)

    map_vals_i = create_mapper_vals_i(map_vals_i)
    map_vals_i = create_mapper_vals_i(feat_arr0.reshape(100, 1))
    map_vals_i = create_mapper_vals_i(None)

    map_vals_i = Map_Vals_i(100)
    map_vals_i = Map_Vals_i((1000, 20))
    map_vals_i = Map_Vals_i(map_vals_i)
    map_vals_i = Map_Vals_i(memb1)

    ## Stress testing
    try:
        boolean = False
        map_vals_i = create_mapper_vals_i('correlation')
        boolean = True
        raise Exception("It has to halt here.")
    except:
        if boolean:
            raise Exception("The test has to halt here.")

    ###########################################################################
    ############################## Spdesc_mapper ##############################
    ###########################################################################
    #selector1 = Sp_DescriptorMapper()
    mapper_array = np.random.randint(0, 5, 100)
    mapper_function = lambda idx: mapper_array[idx]
    mapper_function1 = lambda idx: tuple([mapper_array[idx]]*2)

    pos_mappers = [{'mapper': mapper_array}, {'mapper': mapper_function},
                   {'mapper': mapper_function, 'compute': True},
                   {'mapper': mapper_function, 'n_in': 5, 'n_out': 6},
                   {'mapper': mapper_function1, 'n_in': 5, 'n_out': [3, 4]},
                   {'mapper': mapper_function1, 'n_in': 5, 'compute': True}]

    for p in pos_mappers:
        comb_selector = DummySelector(**p)
#        comb_selector = GeneralSelector(**p)
        comb_selector[0]

        # Impossible cases
        try:
            ## Non-integer key getitem
            boolean = False
            map_vals_i = comb_selector[.2]
            boolean = True
            raise Exception("It has to halt here.")
        except:
            if boolean:
                raise Exception("The test has to halt here.")

        ## Functions
        DummySelector(comb_selector)
        comb_selector[0]
        comb_selector.set_pars(2, lambda x: (0, 0), n_out=[1, 1])
        comb_selector[0]

    selector1 = DummySelector(mapper_array)
    selector2 = DummySelector(lambda idx: mapper_array[idx], n_in=100, n_out=3)
    selector3 = DummySelector(lambda idx: [mapper_array[idx]]*3, n_in=100)
    sl = GeneralCollectionSelectors([selector1, selector2, selector3])

    # Spatial retriever selector
    sel = Spatial_RetrieverSelector(np.array([mapper_array]*2).T)
    sel[0], sel[0, 1], sel[[0, 1]]
    sel = Spatial_RetrieverSelector(mapper_array, mapper_array)
    sel[0], sel[0, 1], sel[[0, 1]]
    sel = Spatial_RetrieverSelector(mapper_function1)
    sel[0], sel[0, 1], sel[[0, 1]]
    sel = Spatial_RetrieverSelector(mapper_function, mapper_function)
    sel[0], sel[0, 1], sel[[0, 1]]
    sel = Spatial_RetrieverSelector(sel)
    sel[0], sel[0, 1], sel[[0, 1]]
    sel = Spatial_RetrieverSelector((0, 0))
    sel[0], sel[0, 1], sel[[0, 1]]
    sel = Spatial_RetrieverSelector(0, 0)
    sel[0], sel[0, 1], sel[[0, 1]]
    try:
        ## Different types of core mappers
        boolean = False
        Spatial_RetrieverSelector(mapper_array, mapper_function)
        boolean = True
        raise Exception("It has to halt here.")
    except:
        if boolean:
            raise Exception("The test has to halt here.")
    try:
        ## Not correct shape
        boolean = False
        Spatial_RetrieverSelector(mapper_array)
        boolean = True
        raise Exception("It has to halt here.")
    except:
        if boolean:
            raise Exception("The test has to halt here.")

    # FeatureInd retriever selector
    sel = FeatInd_RetrieverSelector(np.array([mapper_array]*2).T)
    sel[0], sel[0, 1], sel[[0, 1]]
    sel = FeatInd_RetrieverSelector(mapper_array, mapper_array)
    sel[0], sel[0, 1], sel[[0, 1]]
    sel = FeatInd_RetrieverSelector(mapper_function1)
    sel[0], sel[0, 1], sel[[0, 1]]
    sel = FeatInd_RetrieverSelector(mapper_function, mapper_function)
    sel[0], sel[0, 1], sel[[0, 1]]
    sel = FeatInd_RetrieverSelector(sel)
    sel[0], sel[0, 1], sel[[0, 1]]
    sel = FeatInd_RetrieverSelector((0, 0))
    sel[0], sel[0, 1], sel[[0, 1]]
    sel = FeatInd_RetrieverSelector(0, 0)
    sel[0], sel[0, 1], sel[[0, 1]]
    try:
        ## Different types of core mappers
        boolean = False
        FeatInd_RetrieverSelector(mapper_array, mapper_function)
        boolean = True
        raise Exception("It has to halt here.")
    except:
        if boolean:
            raise Exception("The test has to halt here.")
    try:
        ## Different types of core mappers
        boolean = False
        FeatInd_RetrieverSelector(np.array([mapper_array]*10).T)
        boolean = True
        raise Exception("It has to halt here.")
    except:
        if boolean:
            raise Exception("The test has to halt here.")

    # FeatureInd retriever selector
    sel = Desc_RetrieverSelector(np.array([mapper_array]*2).T)
    sel[0], sel[0, 1], sel[[0, 1]]
    sel = Desc_RetrieverSelector(mapper_array, mapper_array)
    sel[0], sel[0, 1], sel[[0, 1]]
    sel = Desc_RetrieverSelector(mapper_function1)
    sel[0], sel[0, 1], sel[[0, 1]]
    sel = Desc_RetrieverSelector(mapper_function, mapper_function)
    sel[0], sel[0, 1], sel[[0, 1]]
    sel = Desc_RetrieverSelector(sel)
    sel[0], sel[0, 1], sel[[0, 1]]
    sel = Desc_RetrieverSelector((0, 0))
    sel[0], sel[0, 1], sel[[0, 1]]
    sel = Desc_RetrieverSelector(0, 0)
    sel[0], sel[0, 1], sel[[0, 1]]
    try:
        ## Different types of core mappers
        boolean = False
        Desc_RetrieverSelector(mapper_array, mapper_function)
        boolean = True
        raise Exception("It has to halt here.")
    except:
        if boolean:
            raise Exception("The test has to halt here.")
    try:
        ## Different types of core mappers
        boolean = False
        Desc_RetrieverSelector(np.array([mapper_array]*10).T)
        boolean = True
        raise Exception("It has to halt here.")
    except:
        if boolean:
            raise Exception("The test has to halt here.")

    # FeatureInd retriever selector
    pos_selt = [(np.array([mapper_array]*2).T, ), (mapper_array, mapper_array),
                (mapper_function1, ), (mapper_function, mapper_function)]

    def test_getitem(selector):
        selector[0]
        selector[0, 1]
        selector[[0, 1]]
        try:
            boolean = False
            selector[0.7]
            boolean = True
        except:
            if boolean:
                raise Exception("It has to halt here.")

    for i in range(len(pos_selt)):
        ## Instantiation
        sel0 = Spatial_RetrieverSelector(*pos_selt[i])
        sel1 = FeatInd_RetrieverSelector(*pos_selt[i])
        sel2 = FeatInd_RetrieverSelector(*pos_selt[i])
        sel3 = Desc_RetrieverSelector(*pos_selt[i])
        selfeat = Feat_RetrieverSelector(sel1, sel2, sel3)
        test_getitem(selfeat)
        ## Partial information instantiation
        selfeat = Feat_RetrieverSelector(selfeat, None, None)
        test_getitem(selfeat)
        ### Testing Sp_DescriptorSelector
        sel = Sp_DescriptorSelector(sel0, selfeat)
        test_getitem(sel)
        ### Testing Sp_DescriptorSelector with partial instantiation
        sel = Sp_DescriptorSelector(sel)
        test_getitem(sel)

    #### Individual tests
    ## Partial information instantiation
    selfeat = Feat_RetrieverSelector((0, 0, 0, 0, 0, 0), None, None)
    # Getitem
    test_getitem(selfeat)
#    ## Partial information instantiation
    ## Instantiation
    sel1 = FeatInd_RetrieverSelector(sel1)
    sel2 = FeatInd_RetrieverSelector(sel2)
    sel3 = Desc_RetrieverSelector(sel3)
    selfeat = Feat_RetrieverSelector(sel1, sel2, sel3)
    test_getitem(selfeat)
    selfeat = Feat_RetrieverSelector(np.zeros((100, 6)))
    test_getitem(selfeat)
    selfeat = Feat_RetrieverSelector((lambda idx: (0, 0, 0, 0, 0, 0),
                                     {'n_in': 200}))
    test_getitem(selfeat)
