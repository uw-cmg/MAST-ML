__author__ = 'Ryan Jacobs'

from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE

class FeatureSelection(object):
    """Class to conduct feature selection to reduce size of feature space
    """
    def __init__(self, dataframe):
        self.dataframe = dataframe

    @property
    def get_original_dataframe(self):
        return self.dataframe

    def principal_component_analysis(self):
        pass

    def recursive_feature_selection(self):
        pass

