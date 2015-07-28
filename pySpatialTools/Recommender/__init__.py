
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

"""

from pjensen_quality import PjensenRecommender
from neigh_quality import NeighRecommender
from regressors_quality import RandomForestRecommender, KneighRegRecommender,\
	RNeighRegRecommender, GradientBoostingRecommender, ExtraTreesRecommender,\
	AdaBoostRecommender
