__author__ = 'Ryan Jacobs'
__maintainer__ = 'Ryan Jacobs'
__version__ = '1.0'
__email__ = 'rjacobs3@wisc.edu'
__date__ = 'October 14th, 2017'

import pandas as pd
import numpy as np
import sys
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from DataOperations import DataframeUtilities

class FeatureIO(object):
    """
    Class to perform various input/output and manipulations of features present in dataframe

    Args:
        dataframe (pandas dataframe) : dataframe containing x and y data and feature names

    Methods:
        remove_duplicate_features_by_name: DEPRECATED

        remove_duplicate_columns : removes features that have identical names
            Args:
                None
            Returns:
                pandas dataframe : dataframe after feature operation

        remove_duplicate_features_by_values : removes features that have identical values for each data point
            Args:
                None
            Returns:
                pandas dataframe : dataframe after feature operation

        remove_custom_features : remove features as specified by name
            Args:
                features_to_remove (list) : list of feature names to remove
            Returns:
                pandas dataframe : dataframe after feature operation

        keep_custom_features : keep only the specified features
            Args:
                features_to_keep (list) : list of feature names to keep
                y_feature (str) : name of target feature (will keep if specified)
            Returns:
                pandas dataframe : dataframe after feature operation

        add_custom_features : add specific features by name (must also supply data)
            Args:
                features_to_add (list) : list of feature names to add
                data_to_add (numpy array) : array of data for each feature to add
            Returns:
                pandas dataframe : dataframe after feature operation

        custom_feature_filter : removes rows of data if certain arithmetic conditions are met
            Args:
                feature (str) : name of feature to scan data of
                operator (kwarg) : arithmetic operator (choose from <, >, =, <=, >=, <>)
                threshold (float) : value to compare data value against for elimination
            Returns:
                pandas dataframe : dataframe after feature operation
    """
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def remove_duplicate_features_by_name(self, keep=None):
        DeprecationWarning('This method will be removed soon, do not use!')
        # Warning: not sure if this really removes redundant columns. The remove_duplicate_columns method should work, though
        # Only removes features that have the same name, not features containing the same data vector
        dataframe = self.dataframe.drop_duplicates(keep=keep)
        return dataframe

    def remove_duplicate_columns(self):
        # Transferring from dataframe to dict removes redundant columns (i.e. keys of the dict)
        # Do note that the new version of pandas rename duplicate columns when importing with to_csv, so there may be
        # no duplicate columns upon import. However, may still get duplicate columns when manually appending dataframes.
        dataframe_asdict = self.dataframe.to_dict()
        dataframe = pd.DataFrame(dataframe_asdict)
        return dataframe

    def remove_duplicate_features_by_values(self):
        dataframe = self.dataframe.T.drop_duplicates().T
        return dataframe

    def remove_custom_features(self, features_to_remove):
        dataframe = self.dataframe
        if type(features_to_remove) is str:
            features_to_remove_list = list()
            features_to_remove_list.append(features_to_remove)
            features_to_remove = features_to_remove_list
        for feature in features_to_remove:
            del dataframe[feature]
        return dataframe

    def keep_custom_features(self, features_to_keep, y_feature=None):
        dataframe_dict = {}
        if type(features_to_keep) is str:
            features_to_keep_list = list()
            features_to_keep_list.append(features_to_keep)
            features_to_keep = features_to_keep_list
        for feature in features_to_keep:
            dataframe_dict[feature] = self.dataframe[feature]
        if y_feature is not None:
            if y_feature in self.dataframe.columns:
                dataframe_dict[y_feature] = self.dataframe[y_feature]
        dataframe = pd.DataFrame(dataframe_dict)
        return dataframe

    def add_custom_features(self, features_to_add, data_to_add):
        dataframe = self.dataframe
        for feature in features_to_add:
            dataframe[feature] = pd.Series(data=data_to_add, index=self.dataframe.index)
        return dataframe

    def custom_feature_filter(self, feature, operator, threshold):
        # Searches values in feature that meet the condition. If it does, that entire row of data is removed from the dataframe
        rows_to_remove = []
        for i in range(len(self.dataframe[feature])):
            fdata = self.dataframe[feature].iloc[i]
            try:
                fdata = float(fdata)
            except ValueError:
                fdata = fdata
            if operator == '<':
                if fdata < threshold:
                    rows_to_remove.append(i)
            if operator == '>':
                if fdata > threshold:
                    rows_to_remove.append(i)
            if operator == '=':
                if fdata == threshold:
                    rows_to_remove.append(i)
            if operator == '<=':
                if fdata <= threshold:
                    rows_to_remove.append(i)
            if operator == '>=':
                if fdata >= threshold:
                    rows_to_remove.append(i)
            if operator == '<>':
                if not(fdata == threshold):
                    rows_to_remove.append(i)
        dataframe = self.dataframe.drop(self.dataframe.index[rows_to_remove])
        return dataframe

class FeatureNormalization(object):
    """
    Class to normalize and unnormalize features in a dataframe.

    Args
        dataframe (pandas dataframe) : dataframe containing x and y data and feature names

    Methods:
        normalize features : normalizes the specified features to have mean zero and standard deviation of unity
            Args:
                x_features (list) : list of x feature names
                y_feature (str) : target feature name
                normalize_x_features (bool) : whether to normalize x features
                normalize_y_feature (bool) : whether to normalize the target feature
                to_csv (bool) : whether to save normalized dataframe to csv file
            Returns:
                pandas dataframe : dataframe of normalized data
                sklearn scaler object : scaler object used to map unnormalized to normalized data

        minmax_scale_single_feature : scale a single feature to a designated minimum and maximum value
            Args:
                featurename (str) : name of feature to normalize
                smin (float) : minimum feature value
                smax (float) : maximum feature value
            Returns:
                pandas dataframe : dataframe containing only the scaled feature

        unnormalize_features : unnormalize the features contained in a dataframe
            Args:
                x_features (list) : list of x feature names
                y_feature (str) : target feature name
                scaler (sklearn scaler object) : scaler object used to map unnormalized to normalized data
            Returns:
                pandas dataframe : dataframe of unnormalized data
                sklearn scaler object : scaler object used to map unnormalized to normalized data

        normalize_and_merge_with_original_dataframe : normalizes features and merges normalized dataframe with original dataframe
            Args:
                x_features (list) : list of x feature names
                y_feature (str) : target feature name
            Returns:
                pandas dataframe : merged dataframe containing original dataframe and normalized features
    """
    def __init__(self, dataframe, configdict):
        self.dataframe = dataframe
        self.configdict = configdict

    def normalize_features(self, x_features, y_feature, normalize_x_features, normalize_y_feature, feature_normalization_type, feature_scale_min = 0, feature_scale_max = 1, to_csv=True):
        if normalize_x_features == bool(True) and normalize_y_feature == bool(False):
            if feature_normalization_type == 'standardize':
                scaler = StandardScaler().fit(X=self.dataframe[x_features])
            elif feature_normalization_type == 'normalize':
                scaler = MinMaxScaler(feature_range=(feature_scale_min, feature_scale_max)).fit(X=self.dataframe[x_features])
            else:
                print('Error! For feature normalization, you must select either "standardize" or "normalize" for the "feature_normalization_type" option')
                sys.exit()
            array_normalized = scaler.fit_transform(X=self.dataframe[x_features])
            array_normalized = DataframeUtilities().concatenate_arrays(X_array=array_normalized, y_array=np.asarray(self.dataframe[y_feature]).reshape([-1, 1]))
        elif normalize_x_features == bool(False) and normalize_y_feature == bool(True):
            if feature_normalization_type == 'standardize':
                scaler = StandardScaler().fit(X=np.asarray(self.dataframe[y_feature]).reshape([-1, 1]))
            elif feature_normalization_type == 'normalize':
                scaler = MinMaxScaler(feature_range=(feature_scale_min, feature_scale_max)).fit(X=np.asarray(self.dataframe[y_feature]).reshape([-1, 1]))
            else:
                print('Error! For feature normalization, you must select either "standardize" or "normalize" for the "feature_normalization_type" option')
                sys.exit()
            array_normalized = scaler.fit_transform(X=np.asarray(self.dataframe[y_feature]).reshape([-1, 1]))
            array_normalized = DataframeUtilities().concatenate_arrays(X_array=np.asarray(self.dataframe[x_features]), y_array=array_normalized.reshape([-1, 1]))
        elif normalize_x_features == bool(True) and normalize_y_feature == bool(True):
            if feature_normalization_type == 'standardize':
                scaler_x = StandardScaler().fit(X=self.dataframe[x_features])
                scaler_y = StandardScaler().fit(X=np.asarray(self.dataframe[y_feature]).reshape([-1, 1]))
            elif feature_normalization_type == 'normalize':
                scaler_x = MinMaxScaler(feature_range=(feature_scale_min, feature_scale_max)).fit(X=self.dataframe[x_features])
                scaler_y = MinMaxScaler(feature_range=(feature_scale_min, feature_scale_max)).fit(X=np.asarray(self.dataframe[y_feature]).reshape([-1, 1]))
            else:
                print('Error! For feature normalization, you must select either "standardize" or "normalize" for the "feature_normalization_type" option')
                sys.exit()
            array_normalized_x = scaler_x.fit_transform(X=self.dataframe[x_features])
            array_normalized_y = scaler_y.fit_transform(X=np.asarray(self.dataframe[y_feature]).reshape([-1, 1]))
            array_normalized = DataframeUtilities().concatenate_arrays(X_array=array_normalized_x, y_array=array_normalized_y)
        else:
            print("You must specify to normalize either x_features, y_feature, or both, or set perform_feature_normalization=False in the input file")
            sys.exit()

        dataframe_normalized = DataframeUtilities().array_to_dataframe(array=array_normalized)
        dataframe_normalized = DataframeUtilities().assign_columns_as_features(dataframe=dataframe_normalized, x_features=x_features, y_feature=y_feature, remove_first_row=False)
        if to_csv == True:
            # Need configdict to get save path
            #configdict = ConfigFileParser(configfile=sys.argv[1]).get_config_dict(path_to_file=os.getcwd())
            # Get y_feature in this dataframe, attach it to save path
            for column in dataframe_normalized.columns.values:
                if column in self.configdict['General Setup']['target_feature']:
                    filetag = column
            dataframe_normalized.to_csv(self.configdict['General Setup']['save_path']+"/"+'input_data_normalized'+'_'+str(filetag)+'.csv', index=False)

        if not (normalize_x_features == bool(True) and normalize_y_feature == bool(True)):
            return dataframe_normalized, scaler
        else:
            return dataframe_normalized, (scaler_x, scaler_y)

    def minmax_scale_single_feature(self, featurename, smin=None, smax=None):
        feature = self.dataframe[featurename]
        if smin is None:
            smin = np.min(feature)
        if smax is None:
            smax = np.max(feature)
        scaled_feature = (feature - smin) / (smax - smin)
        return scaled_feature

    def unnormalize_features(self, x_features, y_feature, scaler):
        array_unnormalized = scaler.inverse_transform(X=self.dataframe[x_features])
        array_unnormalized = DataframeUtilities().concatenate_arrays(X_array=array_unnormalized, y_array=np.asarray(self.dataframe[y_feature]).reshape([-1, 1]))
        dataframe_unnormalized = DataframeUtilities().array_to_dataframe(array=array_unnormalized)
        dataframe_unnormalized = DataframeUtilities().assign_columns_as_features(dataframe=dataframe_unnormalized, x_features=x_features, y_feature=y_feature, remove_first_row=False)
        return dataframe_unnormalized, scaler

    def normalize_and_merge_with_original_dataframe(self, x_features, y_feature, normalize_x_features, normalize_y_feature):
        dataframe_normalized, scaler = self.normalize_features(x_features=x_features, y_feature=y_feature,
                                                               normalize_x_features=normalize_x_features,
                                                               normalize_y_feature=normalize_y_feature)
        dataframe = DataframeUtilities().merge_dataframe_columns(dataframe1=self.dataframe, dataframe2=dataframe_normalized)
        return dataframe

class MiscFeatureOperations(object):
    """
    Class containing additional feature operations

    Args:
        configdict (dict) : MASTML configfile object as dict

    Methods:
        remove_features_containing_strings : removes feature columns whose values are strings as these can't be used in regression tasks
            Args:
                dataframe (pandas dataframe) : dataframe containing data and feature names
                x_features (list) : list of x feature names
            Returns:
                list : list of x features with those features removed which contained data as strings
                pandas dataframe : dataframe containing data and feature names, with string features removed
    """
    def __init__(self, configdict):
        self.configdict = configdict

    def remove_features_containing_strings(self, dataframe, x_features):
        x_features_pruned = []
        x_features_to_remove = []
        for x_feature in x_features:
            is_str = False
            for entry in dataframe[x_feature]:
                if type(entry) is str:
                    is_str = True
            if is_str == True:
                if 'grouping_feature' in self.configdict['General Setup'].keys():
                    if x_feature not in self.configdict['General Setup']['grouping_feature']:
                        x_features_to_remove.append(x_feature)
                else:
                    x_features_to_remove.append(x_feature)
        for x_feature in x_features:
            if x_feature not in x_features_to_remove:
                x_features_pruned.append(x_feature)

        dataframe_pruned = FeatureIO(dataframe=dataframe).remove_custom_features(features_to_remove=x_features_to_remove)
        return x_features_pruned, dataframe_pruned