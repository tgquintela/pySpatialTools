
"""
Supervised Recommender Model
----------------------------
Module which contains the classes and needed functions for the computation of
recommender models based on supervised machine learning techniques.
"""

class SupervisedRmodel(RecommenderModel):
	"Generic abstract class for the supervised models "

	def __init__(self, skmodel):
		self.model = skmodel

	def fit_model(self, descriptor_matrix, correctness):
		pass

	def compute_quality(self, descriptor_matrix):
		pass
