__author__ = 'Ryan Jacobs'

import pandas as pd
import logging
import sys
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from pymatgen import Element, Composition
from sklearn.preprocessing import minmax_scale
#TTM install error - commenting out for now
#from matminer.descriptors.composition_features import get_magpie_descriptor


class DataParser(object):
    """Class to parse input csv file and create pandas dataframe, and extract features
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
            dataframe = pd.read_csv(datapath, header=0)
            #column_names = dataframe.iloc[0].tolist()
        except IOError:
            logging.info('Error reading in your input data file, specify a valid path to your input data')
            sys.exit()
        return dataframe

    def get_features(self, dataframe, target_feature=None, from_input_file=False):
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
        if not(y_feature in dataframe.columns):
            logging.warning("%s not in columns" % y_feature)
            ydata = None
        else:
            ydata = dataframe.loc[:, y_feature]
        return Xdata, ydata

class FeatureIO(object):
    """Class to selectively filter (add/remove) features from a dataframe
    """
    def __init__(self, dataframe):
        self.dataframe = dataframe

    @property
    def get_original_dataframe(self):
        return self.dataframe

    def remove_duplicate_features_by_name(self):
        # Only removes features that have the same name, not features containing the same data vector
        dataframe = self.dataframe.drop_duplicates()
        return dataframe

    def remove_custom_features(self, features_to_remove):
        dataframe = self.dataframe
        for feature in features_to_remove:
            del dataframe[feature]
        return dataframe

    def keep_custom_features(self, features_to_keep, y_feature):
        dataframe_dict = {}
        for feature in features_to_keep:
            dataframe_dict[feature] = self.dataframe[feature]
        dataframe_dict[y_feature] = self.dataframe[y_feature]
        dataframe = pd.DataFrame(dataframe_dict)
        return dataframe

    def add_custom_features(self, features_to_add, data_to_add):
        dataframe = self.dataframe
        for feature in features_to_add:
            dataframe[feature] = pd.Series(data=data_to_add, index=(self.dataframe).index)
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
    """This class is used to normalize and unnormalize features in a dataframe.
    """
    def __init__(self, dataframe):
        self.dataframe = dataframe

    @property
    def get_original_dataframe(self):
        return self.dataframe

    def normalize_features(self, x_features, y_feature):
        scaler = StandardScaler().fit(X=self.dataframe[x_features])
        array_normalized = scaler.fit_transform(X=self.dataframe[x_features], y=self.dataframe[y_feature])
        array_normalized = DataframeUtilities()._concatenate_arrays(X_array=array_normalized, y_array=np.asarray(self.dataframe[y_feature]).reshape([-1, 1]))
        dataframe_normalized = DataframeUtilities()._array_to_dataframe(array=array_normalized)
        dataframe_normalized = DataframeUtilities()._assign_columns_as_features(dataframe=dataframe_normalized, x_features=x_features, y_feature=y_feature, remove_first_row=False)
        return dataframe_normalized, scaler

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
        array_unnormalized = DataframeUtilities()._concatenate_arrays(X_array=array_unnormalized, y_array=np.asarray(self.dataframe[y_feature]).reshape([-1, 1]))
        dataframe_unnormalized = DataframeUtilities()._array_to_dataframe(array=array_unnormalized)
        dataframe_unnormalized = DataframeUtilities()._assign_columns_as_features(dataframe=dataframe_unnormalized, x_features=x_features, y_feature=y_feature, remove_first_row=False)
        return dataframe_unnormalized, scaler

    def normalize_and_merge_with_original_dataframe(self, x_features, y_feature):
        dataframe_normalized, scaler = self.normalize_features(x_features=x_features, y_feature=y_feature)
        dataframe = DataframeUtilities()._merge_dataframe_columns(dataframe1=self.dataframe, dataframe2=dataframe_normalized)
        return dataframe

class MagpieFeatures(object):
    """Class to generate new features using Magpie data and dataframe containing material compositions. Creates
     a dataframe and append features to existing feature dataframes
    """
    def __init__(self, dataframe):
        self.dataframe = dataframe

    @property
    def get_original_dataframe(self):
        return self.dataframe

    def generate_magpie_features(self, save_to_csv=True):
        try:
            compositions = self.dataframe['Magpie compositions']
        except KeyError:
            print('No column called "Magpie compositions" exists in the supplied dataframe.')
            sys.exit()

        magpiedata_dict_composition_average = {}
        magpiedata_dict_arithmetic_average = {}
        for composition in compositions:
            magpiedata_composition_average, magpiedata_arithmetic_average = self._get_computed_magpie_features(composition=composition)
            magpiedata_dict_composition_average[composition] = magpiedata_composition_average
            magpiedata_dict_arithmetic_average[composition] = magpiedata_arithmetic_average

        magpiedata_dict_list = [magpiedata_dict_composition_average, magpiedata_dict_arithmetic_average]
        dataframe = self.dataframe
        for magpiedata_dict in magpiedata_dict_list:
            dataframe_magpie = pd.DataFrame.from_dict(data=magpiedata_dict, orient='index')
            # Need to reorder compositions in new dataframe to match input dataframe
            dataframe_magpie = dataframe_magpie.reindex(self.dataframe['Magpie compositions'].tolist())
            # Need to make compositions the first column, instead of the row names
            dataframe_magpie.index.name = 'Magpie compositions'
            dataframe_magpie.reset_index(inplace=True)
            # Need to delete duplicate column before merging dataframes
            del dataframe_magpie['Magpie compositions']
            # Merge magpie feature dataframe with originally supplied dataframe
            dataframe = DataframeUtilities()._merge_dataframe_columns(dataframe1=dataframe, dataframe2=dataframe_magpie)
        if save_to_csv == bool(True):
            dataframe.to_csv('input_with_magpie_features.csv', index=False)
        return dataframe

    def _get_atomic_magpie_features(self, composition):
        # Get .table files containing feature values for each element, assign file names as feature names
        data_path = './magpiedata/magpie_elementdata'
        magpie_feature_names = []
        for f in os.listdir(data_path):
            if '.table' in f:
                magpie_feature_names.append(f[:-6])

        composition = Composition(composition)
        element_list, atoms_per_formula_unit = self._get_element_list(composition=composition)

        element_dict = {}
        for element in element_list:
            element_dict[element] = Element(element).Z

        magpiedata_atomic = {}
        for k, v in element_dict.items():
            atomic_values ={}
            for feature_name in magpie_feature_names:
                f = open(data_path + '/' + feature_name + '.table', 'r')
                # Get Magpie data of relevant atomic numbers for this composition
                for line, feature_value in enumerate(f.readlines()):
                    if line + 1 == v:
                        if "Missing" not in feature_value and "NA" not in feature_value:
                            if feature_name != "OxidationStates":
                                atomic_values[feature_name] = float(feature_value.strip())
                        if "Missing" in feature_value:
                            atomic_values[feature_name] = 'NaN'
                        if "NA" in feature_value:
                            atomic_values[feature_name] = 'NaN'
                f.close()
            magpiedata_atomic[k] = atomic_values

        return magpiedata_atomic

    def _get_computed_magpie_features(self, composition):
        magpiedata_composition_average = {}
        magpiedata_arithmetic_average = {}
        magpiedata_max = {}
        magpiedata_min = {}
        magpiedata_atomic = self._get_atomic_magpie_features(composition=composition)
        composition = Composition(composition)
        element_list, atoms_per_formula_unit = self._get_element_list(composition=composition)

        # Initialize feature values to all be 0, because need to dynamically update them with weighted values in next loop.
        for magpie_feature in magpiedata_atomic[element_list[0]].keys():
            magpiedata_composition_average[magpie_feature] = 0
            magpiedata_arithmetic_average[magpie_feature] = 0
            magpiedata_max[magpie_feature] = 0
            magpiedata_min[magpie_feature] = 0

        for element in magpiedata_atomic.keys():
            for magpie_feature, feature_value in magpiedata_atomic[element].items():
                if feature_value is not 'NaN':
                    # Composition average features
                    magpiedata_composition_average[magpie_feature] += feature_value*float(composition[element])/atoms_per_formula_unit
                    # Arithmetic average features
                    magpiedata_arithmetic_average[magpie_feature] += feature_value/len(element_list)
                    # Max features
                    if magpiedata_max[magpie_feature] > 0:
                        if feature_value > magpiedata_max[magpie_feature]:
                            magpiedata_max[magpie_feature] = feature_value
                    elif magpiedata_max[magpie_feature] == 0:
                        magpiedata_max[magpie_feature] = feature_value
                    # Min features
                    if magpiedata_min[magpie_feature] > 0:
                        if feature_value < magpiedata_min[magpie_feature]:
                            magpiedata_min[magpie_feature] = feature_value
                    elif magpiedata_min[magpie_feature] == 0:
                        magpiedata_min[magpie_feature] = feature_value
                if feature_value is 'NaN':
                    magpiedata_composition_average[magpie_feature] = 'NaN'

        # Change names of features to reflect each computed type of magpie feature (max, min, etc.)
        magpiedata_composition_average_renamed = {}
        magpiedata_arithmetic_average_renamed = {}
        for key in magpiedata_composition_average.keys():
            magpiedata_composition_average_renamed[key+"_composition_average"] = magpiedata_composition_average[key]
        for key in magpiedata_arithmetic_average.keys():
            magpiedata_arithmetic_average_renamed[key+"_arithmetic_average"] = magpiedata_arithmetic_average[key]

        return magpiedata_composition_average_renamed, magpiedata_arithmetic_average_renamed

    def _get_element_list(self, composition):
        element_amounts = composition.get_el_amt_dict()
        atoms_per_formula_unit = 0
        for v in element_amounts.values():
            atoms_per_formula_unit += v

        # Get list of unique elements present
        element_list = []
        for k in element_amounts.keys():
            if k not in element_list:
                element_list.append(k)

        return element_list, atoms_per_formula_unit

class DataframeUtilities(object):
    """This class is a collection of basic utilities for dataframe manipulation, and exchanging between dataframes and numpy arrays
    """
    @classmethod
    def _merge_dataframe_columns(cls, dataframe1, dataframe2):
        dataframe = pd.concat([dataframe1, dataframe2], axis=1)
        return dataframe

    @classmethod
    def _merge_dataframe_rows(cls, dataframe1, dataframe2):
        dataframe = pd.merge(left=dataframe1, right=dataframe2, how='outer')
        return dataframe

    @classmethod
    def _get_dataframe_statistics(cls, dataframe):
        return dataframe.describe(include='all')

    @classmethod
    def _dataframe_to_array(cls, dataframe):
        array = np.asarray(dataframe)
        return array

    @classmethod
    def _array_to_dataframe(cls, array):
        dataframe = pd.DataFrame(data=array, index=range(1, len(array)+1))
        return dataframe

    @classmethod
    def _concatenate_arrays(cls, X_array, y_array):
        array = np.concatenate((X_array, y_array), axis=1)
        return array

    @classmethod
    def _assign_columns_as_features(cls, dataframe, x_features, y_feature, remove_first_row=True):
        column_dict = {}
        x_and_y_features = [feature for feature in x_features]
        x_and_y_features.append(y_feature)
        for i, feature in enumerate(x_and_y_features):
            column_dict[i] = feature
        dataframe = dataframe.rename(columns=column_dict)
        if remove_first_row == bool(True):
            dataframe = dataframe.drop([0])  # Need to remove feature names from first row so can obtain data
        return dataframe
