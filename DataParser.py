__author__ = 'Ryan Jacobs'

import pandas as pd
import logging
import sys
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class DataParser(object):
    """Class to parse input csv file and create pandas dataframe, and extract features
    """
    def __init__(self, configdict):
        self.configdict = configdict

    def parse_fromfile(self, datapath, as_array=False):

        dataframe = self.import_data(datapath=datapath)
        x_features, y_feature = self.get_features(dataframe=dataframe, target_feature=None, from_input_file=True)
        fops = FeatureOperations(dataframe=dataframe)
        dataframe = fops.assign_columns_as_features(x_features=x_features, y_feature=y_feature)
        Xdata, ydata = self.get_data(dataframe=dataframe, x_features=x_features, y_feature=y_feature)

        if as_array == bool(True):
            Xdata = np.asarray(Xdata)
            ydata = np.asarray(ydata)

        return Xdata, ydata, x_features, y_feature, dataframe

    def parse_fromdataframe(self, dataframe, target_feature, as_array=False):
        x_features, y_feature = self.get_features(dataframe=dataframe, target_feature=target_feature, from_input_file=False)
        Xdata, ydata = self.get_data(dataframe=dataframe, x_features=x_features, y_feature=y_feature)

        if as_array == bool(True):
            Xdata = np.asarray(Xdata)
            ydata = np.asarray(ydata)

        return Xdata, ydata, x_features, y_feature, dataframe

    def import_data(self, datapath):
        try:
            dataframe = pd.read_csv(datapath, header=None)
        except IOError:
            logging.info('Error reading in your input data file, specify a valid path to your input data')
            sys.exit()
        return dataframe

    def get_features(self, dataframe, target_feature=None, from_input_file=True):
        if from_input_file == bool(True):
            y_feature = self.configdict['General Setup']['target_feature']
            if self.configdict['General Setup']['input_features'] == ['Auto']:
                x_and_y_features = dataframe.loc[0, :].tolist()
                x_features = []
                for feature in x_and_y_features:
                    if feature != y_feature:
                        x_features.append(feature)
            else:
                x_features = [feature for feature in self.configdict['General Setup']['input_features']]
        elif from_input_file == bool(False):
            y_feature = target_feature
            x_features = [feature for feature in dataframe.columns.values if feature != y_feature]

        return x_features, y_feature

    def get_data(self, dataframe, x_features, y_feature):
        Xdata = dataframe.loc[:, x_features]
        ydata = dataframe.loc[:, y_feature]
        return Xdata, ydata

class FeatureOperations(object):
    """Class to selectively filter features from a dataframe
    """
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def remove_duplicate_features(self):
        # Only removes features that have the same name, not features containing the same data vector
        (self.dataframe).drop_duplicates()
        return self.dataframe

    def remove_custom_features(self, features_to_remove):
        for feature in features_to_remove:
            del self.dataframe[feature]
        return self.dataframe

    def add_custom_features(self, features_to_add, data_to_add):
        for feature in features_to_add:
            self.dataframe[feature] = pd.Series(data=data_to_add, index=(self.dataframe).index)
        return self.dataframe

    def assign_columns_as_features(self, x_features, y_feature, remove_first_row=True):
        column_dict = {}
        x_and_y_features = [feature for feature in x_features]
        x_and_y_features.append(y_feature)
        for i, feature in enumerate(x_and_y_features):
            column_dict[i] = feature
        dataframe = self.dataframe.rename(columns=column_dict)
        if remove_first_row == bool(True):
            dataframe = dataframe.drop([0])  # Need to remove feature names from first row so can obtain data
        return dataframe

class DataOperations(object):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    # This method may not be needed as PCA and feature selection may make it obsolete
    def remove_duplicate_data(self, x_features, y_feature):
        # WARNING: this function currently doesn't work. Still looking into this.
        print(x_features, y_feature)
        selector = VarianceThreshold(threshold = 0)
        array = selector.fit_transform(X=self.dataframe[x_features], y=self.dataframe[y_feature])
        y_data = np.asarray(self.dataframe[y_feature]).reshape([-1, 1])
        data = np.concatenate((array, y_data), axis=1)
        print(data)
        dataframe = pd.DataFrame(data=data)
        fops = FeatureOperations(dataframe=dataframe)
        print(x_features, y_feature)
        dataframe = fops.assign_columns_as_features(x_features=x_features, y_feature=y_feature)
        return dataframe

    def merge_dataframes(self, dataframe_to_merge):
        dataframe = pd.merge(left=self.dataframe, right=dataframe_to_merge, how='inner')
        return dataframe

    def dataframe_statistics(self):
        return (self.dataframe).describe(include='all')

    def normalize_data(self):
        pass

    def unnormalize_data(self):
        pass


