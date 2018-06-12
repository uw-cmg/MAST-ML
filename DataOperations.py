__author__ = 'Ryan Jacobs'
__maintainer__ = 'Ryan Jacobs'
__version__ = '1.0'
__email__ = 'rjacobs3@wisc.edu'
__date__ = 'October 14th, 2017'

import pandas as pd
import logging
import sys
import numpy as np
import os
from matplotlib import pyplot as plt
from MASTMLInitializer import ConfigFileParser

class DataParser(object):
    """
    Class to parse input csv file and create pandas dataframe, and extract features and data

    Args:
        configdict (dict) : MASTML configfile object as dict

    Methods:
        parse_fromfile : parses data and features from input file

            Args:
                datapath (str) : path of location of input data csv file
                as_array (bool) : whether to return data in form of numpy array

            Returns:
                pandas dataframe or numpy array : feature data matrix
                pandas dataframe or numpy array : target data vector
                list : list of x feature names
                str : target feature name
                pandas dataframe : dataframe containing data and feature names

        parse_fromdataframe : parses data and features from pandas dataframe

            Args:
                dataframe (pandas dataframe) : dataframe containing data and feature names
                target_feature (str) : target feature name
                as_array (bool) : whether to return data in form of numpy array

            Returns:
                pandas dataframe or numpy array : feature data matrix
                pandas dataframe or numpy array : target data vector
                list : list of x feature names
                str : target feature name
                pandas dataframe : dataframe containing data and feature names

        import_data : reads in csv file from supplied data path

            Args:
                datapath (str) : string of path to csv file

            Returns:
                pandas dataframe : dataframe representation of csv contents

        get_features : obtains x and y features of input data based on user-supplied input file

            Args:
                dataframe (pandas dataframe) : dataframe representation of csv contents
                target_feature (str) : target feature name
                from_input_file (bool) : whether to read-in data from input file path. If False, reads from dataframe

            Returns:
                list : list of x feature names
                str : target feature name

        get_data : obtains X and y data from dataframe

            Args:
                dataframe (pandas dataframe) : a pandas dataframe object containg X and y data to read in
                x_features (list) : list of x feature names
                y_feature (str) : target feature name

            Returns:
                pandas dataframe : dataframe of x data only
                pandas dataframe : dataframe of y data only
    """
    def __init__(self, configdict=None):
        self.configdict = configdict

    def parse_fromfile(self, datapath, as_array=False):
        if self.configdict is not None:
            dataframe = self.import_data(datapath=datapath)
            x_features, y_feature = self.get_features(dataframe=dataframe, target_feature=None, from_input_file=True)
            #dataframe = DataframeUtilities()._assign_columns_as_features(dataframe=dataframe, x_features=x_features, y_feature=y_feature, column_names=column_names, remove_first_row=True)
            Xdata, ydata = self.get_data(dataframe=dataframe, x_features=x_features, y_feature=y_feature)

            if as_array == bool(True):
                Xdata = np.asarray(Xdata)
                ydata = np.asarray(ydata)
        else:
            raise OSError('You must specify a configdict as input to use the parse_fromfile method')

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
            if '.csv' in datapath.split('/')[-1]:
                dataframe = pd.read_csv(datapath, header=0)
            elif '.xlsx' or '.xls' in datapath.split('/')[-1]:
                dataframe = pd.read_excel(datapath, header=0)
        except IOError:
            logging.info('Error reading in your input data file, specify a valid path to your input data')
            sys.exit()
        return dataframe

    def get_features(self, dataframe, target_feature=None, from_input_file=False):
        if from_input_file == bool(True):
            y_feature_from_input = self.configdict['General Setup']['target_feature']

            x_and_y_features = dataframe.columns.values.tolist()
            if self.configdict['General Setup']['input_features'] == ['Auto'] or self.configdict['General Setup']['input_features'] == 'Auto':
                logging.info('Found Auto')
                #x_and_y_features = dataframe.columns.values.tolist()
                x_features = []
                for feature in x_and_y_features:
                    if feature not in y_feature_from_input:
                        if 'grouping_feature' in self.configdict['General Setup'].keys():
                            if feature not in self.configdict['General Setup']['grouping_feature']:
                                x_features.append(feature)
                        else:
                            x_features.append(feature)
            else:
                x_features = [feature for feature in self.configdict['General Setup']['input_features']]

            for feature in x_and_y_features:
                if feature not in x_features:
                    if feature in y_feature_from_input:
                        y_feature = feature

        elif from_input_file == bool(False):
            y_feature = target_feature
            if 'grouping_feature' in self.configdict['General Setup'].keys():
                x_features = [feature for feature in dataframe.columns.values if feature not in [y_feature, self.configdict['General Setup']['grouping_feature']]]
            else:
                x_features = [feature for feature in dataframe.columns.values if feature not in y_feature]

        try:
            y_feature
        except UnboundLocalError:
            y_feature = self.configdict['General Setup']['target_feature']
            dataframe[y_feature] = np.zeros(shape=dataframe.shape[0])
            logging.info('MASTML has detected that a data file has been supplied that does not contain the designated '
                         'y_feature name (probably because you are doing a PREDICTION run)! By default, this feature '
                         'column has been assigned a value of 0 for each data instance (as the true values are not known). '
                         'Mind that the SingleFit and residuals plot for your prediction will be meaningless, but your predicted '
                         'values are still accurate!')

        return x_features, y_feature

    def get_data(self, dataframe, x_features, y_feature):
        Xdata = dataframe.loc[:, x_features]
        ydata = dataframe.loc[:, y_feature]
        return Xdata, ydata

class DataframeUtilities(object):
    """
    Class of basic utilities for dataframe manipulation, and exchanging between dataframes and numpy arrays

    Methods:
        merge_dataframe_columns : merge two dataframes by concatenating the column names (duplicate columns omitted)

            Args:
                dataframe1 (pandas dataframe) : a pandas dataframe object
                dataframe2 (pandas dataframe) : a pandas dataframe object

            Returns:
                pandas dataframe : merged dataframe

        merge_dataframe_rows : merge two dataframes by concatenating the row contents (duplicate rows omitted)

            Args:
                dataframe1 (pandas dataframe) : a pandas dataframe object
                dataframe2 (pandas dataframe) : a pandas dataframe object

            Returns:
                pandas dataframe : merged dataframe

        get_dataframe_statistics : obtain basic statistics about data contained in the dataframe

            Args:
                dataframe (pandas dataframe) : a pandas dataframe object

            Returns:
                pandas dataframe : dataframe containing input dataframe statistics

        dataframe_to_array : transform a pandas dataframe to a numpy array

            Args:
                dataframe (pandas dataframe) : a pandas dataframe object

            Returns:
                numpy array : a numpy array representation of the inputted dataframe

        array_to_dataframe : transform a numpy array to a pandas dataframe

            Args:
                array (numpy array) : a numpy array object

            Returns:
                pandas dataframe : a pandas dataframe representation of the inputted numpy array

        concatenate_arrays : merge two numpy arrays by concatenating along the columns

            Args:
                Xarray (numpy array) : a numpy array object
                yarray (numpy array) : a numpy array object

            Returns:
                numpy array : a numpy array merging the two input arrays

        assign_columns_as_features : adds column names to dataframe based on the x and y feature names

            Args:
                dataframe (pandas dataframe) : a pandas dataframe object
                x_features (list) : list containing x feature names
                y_feature (str) : target feature name

            Returns:
                pandas dataframe : dataframe containing same data as input, with columns labeled with features

        save_all_dataframe_statistics : obtain dataframe statistics and save it to a csv file

            Args:
                dataframe (pandas dataframe) : a pandas dataframe object
                data_path (str) : file path to save dataframe statistics to

            Returns:
                str : name of file dataframe stats saved to

        plot_dataframe_histogram : creates a histogram plot of target feature data and saves it to designated save path

            Args:
                configdict (dict) : MASTML configfile object as dict
                dataframe (pandas dataframe) : a pandas dataframe object
                y_feature (str) : target feature name

            Returns:
                str : name of file dataframe histogram saved to
    """
    @classmethod
    def merge_dataframe_columns(cls, dataframe1, dataframe2):
        dataframe = pd.concat([dataframe1, dataframe2], axis=1, join='outer')
        return dataframe

    @classmethod
    def merge_dataframe_rows(cls, dataframe1, dataframe2):
        dataframe = pd.merge(left=dataframe1, right=dataframe2, how='outer')
        #dataframe = pd.concat([dataframe1, dataframe2], axis=1, join='outer')
        return dataframe

    @classmethod
    def get_dataframe_statistics(cls, dataframe):
        dataframe_stats = dataframe.describe(include='all')
        return dataframe_stats

    @classmethod
    def dataframe_to_array(cls, dataframe):
        array = np.asarray(dataframe)
        return array

    @classmethod
    def array_to_dataframe(cls, array):
        dataframe = pd.DataFrame(data=array, index=range(0, len(array)))
        return dataframe

    @classmethod
    def concatenate_arrays(cls, X_array, y_array):
        array = np.concatenate((X_array, y_array), axis=1)
        return array

    @classmethod
    def remove_dataframe_rows_by_index(cls, dataframe, rows):
        dataframe = dataframe.drop(dataframe.index[rows])
        return dataframe

    @classmethod
    def get_dataframe_nan_indices(cls, dataframe):
        df_isna = pd.isna(dataframe)
        na_indices = [i for i in df_isna.index.values if df_isna[i] == True]
        return na_indices

    @classmethod
    def assign_columns_as_features(cls, dataframe, x_features, y_feature, remove_first_row=True):
        column_dict = {}
        x_and_y_features = [feature for feature in x_features]
        x_and_y_features.append(y_feature)
        for i, feature in enumerate(x_and_y_features):
            column_dict[i] = feature
        dataframe = dataframe.rename(columns=column_dict)
        if remove_first_row == bool(True):
            dataframe = dataframe.drop([0])  # Need to remove feature names from first row so can obtain data
        return dataframe

    @classmethod
    def save_all_dataframe_statistics(cls, dataframe, configdict):
        dataframe_stats = cls.get_dataframe_statistics(dataframe=dataframe)
        data_path_name = configdict['General Setup']['target_feature']
        fname = configdict['General Setup']['save_path'] + "/" + 'input_data_statistics_'+data_path_name+'.csv'
        dataframe_stats.to_csv(fname, index=True)
        return fname

    @classmethod
    def plot_dataframe_histogram(cls, dataframe, title, xlabel, ylabel, save_path, file_name):
        na_indices = cls.get_dataframe_nan_indices(dataframe=dataframe)
        df_y = cls.remove_dataframe_rows_by_index(dataframe=dataframe, rows=na_indices)
        num_bins = cls.get_histogram_bins(dataframe=dataframe)
        plt.figure()
        plt.hist(x=df_y, bins=num_bins, edgecolor='k')
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        fpath = save_path+'/'+file_name
        plt.tight_layout()
        plt.savefig(fpath)
        return fpath

    @classmethod
    def get_histogram_bins(cls, dataframe):
        bin_dividers = np.linspace(dataframe.shape[0], round(0.05*dataframe.shape[0]), dataframe.shape[0])
        bin_list = list()
        for divider in bin_dividers:
            bins = int((dataframe.shape[0])/divider)
            if bins < dataframe.shape[0]/2:
                bin_list.append(bins)
        if len(bin_list) > 0:
            num_bins = max(bin_list)
        else:
            num_bins = 10
        return num_bins