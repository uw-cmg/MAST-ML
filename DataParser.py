__author__ = 'Ryan Jacobs'

import pandas as pd
import logging
import sys
import numpy as np

class InputDataParser(object):
    """Class to parse input csv file and create pandas dataframe, and extract features
    """
    def __init__(self, datapath, configdict, dataframe=None, as_array=False):
        self.datapath = datapath
        self.configdict = configdict
        self.dataframe = dataframe
        self.as_array = as_array

    def parse(self):
        if self.dataframe is None:
            dataframe = self.import_data()
            x_features, y_feature = self.get_features(dataframe=dataframe)
            dataframe = self.assign_columns_as_features(dataframe=dataframe, x_features=x_features, y_feature=y_feature)
            Xdata, ydata = self.get_data(dataframe=dataframe, x_features=x_features, y_feature=y_feature)
        else:
            x_features, y_feature = self.get_features(dataframe=self.dataframe)
            Xdata, ydata = self.get_data(dataframe=self.dataframe, x_features=x_features, y_feature=y_feature)

        if self.as_array == bool(True):
            Xdata = np.asarray(Xdata)
            ydata = np.asarray(ydata)

        return Xdata, ydata, x_features, y_feature

    def import_data(self):
        try:
            dataframe = pd.read_csv(self.datapath, header=None)
        except IOError:
            logging.info('Error reading in your input data file, specify a valid path to your input data')
            sys.exit()
        return dataframe

    def get_features(self, dataframe):
        y_feature = self.configdict['General Setup']['target_feature']
        if self.configdict['General Setup']['input_features'] == ['Auto']:
            x_and_y_features = dataframe.loc[0, :].tolist()
            x_features = []
            for feature in x_and_y_features:
                if feature != y_feature:
                    x_features.append(feature)
        else:
            x_features = [feature for feature in self.configdict['General Setup']['input_features']]

        return x_features, y_feature

    @staticmethod
    def assign_columns_as_features(dataframe, x_features, y_feature):
        column_dict = {}
        x_and_y_features = [feature for feature in x_features]
        x_and_y_features.append(y_feature)
        for i, feature in enumerate(x_and_y_features):
            column_dict[i] = feature
        dataframe = dataframe.rename(columns=column_dict)
        dataframe = dataframe.drop([0]) # Need to remove feature names from first row so can obtain data
        return dataframe

    def get_data(self, dataframe, x_features, y_feature):
        Xdata = dataframe.loc[:, x_features]
        ydata = dataframe.loc[:, y_feature]
        return Xdata, ydata


class FeatureFilter(InputDataParser):
    """Class to selectively filter features from a dataframe
    """
    def __init__(self, datapath, configdict, dataframe, as_array):
        super().__init__(datapath, configdict, dataframe, as_array)

    def remove_features(self, features_to_remove):
        for feature in features_to_remove:
            del self.dataframe[feature]
        return self.dataframe


class DataNormalization(InputDataParser):
    def __init__(self, datapath, configdict):
        super().__init__(datapath, configdict)
