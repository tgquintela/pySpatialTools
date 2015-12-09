
"""
Recommender
===========
Package which contains the classes needed to recommend place from descriptors
and models.

TODO
----
- Classifiers
- Incorporate the testers in the class
- Creation of plots in the class


Structure
---------
 - RecommenderModel
    |
    | - CorrRecomender
    | - MLRecomender

"""
#from recommender_model import RecommenderModel
from pjensen_quality import PjensenRecommender
from neigh_quality import NeighRecommender
#from regressors_quality import RandomForestRecommender, KneighRegRecommender,\
#    RNeighRegRecommender, GradientBoostingRecommender, ExtraTreesRecommender,\
#    AdaBoostRecommender

from supervisedmodels import SupervisedRmodel

## Test measures
from tester_recommender import binary_categorical_measure,\
    float_categorical_measure, binary_quality_measure, float_quality_measure
