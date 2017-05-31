__author__ = 'Ryan Jacobs'

from sklearn.decomposition import PCA
from DataOperations import DataframeUtilities
from FeatureOperations import FeatureIO
from sklearn.feature_selection import RFE
from sklearn.feature_selection import VarianceThreshold

class FeatureSelection(object):
    """Class to conduct feature selection to reduce size of feature space
    """
    def __init__(self, dataframe):
        self.dataframe = dataframe

    @property
    def get_original_dataframe(self):
        return self.dataframe

    def principal_component_analysis(self, x_features, y_feature):
        pca = PCA(n_components=len(x_features), svd_solver='auto')
        Xnew = pca.fit_transform(X=self.dataframe[x_features])
        dataframe = DataframeUtilities()._array_to_dataframe(array=Xnew)
        dataframe = FeatureIO(dataframe=dataframe).add_custom_features(features_to_add=[y_feature], data_to_add=self.dataframe[y_feature])
        return dataframe

    def recursive_feature_selection(self):
        pass

