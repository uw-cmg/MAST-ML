"""
A collection of classes for generating extra features.
All classes here assume dataframe input and guarantee dataframe output.
(So no numpy arrays.)
"""

# TODO: implement API feature generation!!

import multiprocessing
import os
import sys
import warnings

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PolynomialFeatures as SklearnPolynomialFeatures

from pymatgen import Element, Composition
from pymatgen.ext.matproj import MPRester
from citrination_client import CitrinationClient, PifQuery, SystemQuery, ChemicalFieldQuery, ChemicalFilter
# trouble? try: `pip install citrination_client=="2.1.0"`

import mastml
from .util_legos import DoNothing

# locate path to directory containing AtomicNumber.table, AtomicRadii.table AtomicVolume.table, etc
# needs to do it the hard way becuase python -m sets cwd to wherever python is ran from.
print('mastml dir: ', mastml.__path__)
MAGPIE_DATA_PATH = os.path.join(mastml.__path__[0], '../magpie/')


class PolynomialFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, features=None, degree=2, interaction_only=False, include_bias=True):
        self.features = features
        self.SPF = SklearnPolynomialFeatures(degree, interaction_only, include_bias)
    def fit(self, df, y=None):
        if self.features is None:
            self.features = df.columns
        array = df[self.features].values
        self.SPF.fit(array)
        return self
    def transform(self, df):
        array = df[self.features].values
        return pd.DataFrame(self.SPF.transform(array))


class Magpie(BaseEstimator, TransformerMixin):
    def __init__(self, composition_feature):
        self.composition_feature = composition_feature

    def fit(self, df, y=None):
        self.original_features = df.columns
        return self

    def transform(self, df):
        mfg = MagpieFeatureGeneration(df, self.composition_feature)
        df = mfg.generate_magpie_features()

        df = df.drop(self.original_features, axis=1)
        # delete missing values, generation makes a lot of garbage.
        df = clean_dataframe(df)
        df = df.select_dtypes(['number']).dropna(axis=1)
        assert self.composition_feature not in df.columns
        return df


class MaterialsProject(BaseEstimator, TransformerMixin):
    def __init__(self, composition_feature, api_key):
        self.composition_feature = composition_feature
        self.api_key = api_key

    def fit(self, df, y=None):
        self.original_features = df.columns
        return self

    def transform(self, df):
        # make materials project api call (uses internet)
        mpg = MaterialsProjectFeatureGeneration(df.copy(), self.api_key, self.composition_feature)
        df = mpg.generate_materialsproject_features()

        df = df.drop(self.original_features, axis=1)
        # delete missing values, generation makes a lot of garbage.
        df = clean_dataframe(df)
        assert self.composition_feature not in df.columns
        return df

class Citrine(BaseEstimator, TransformerMixin):
    def __init__(self, composition_feature, api_key):
        self.composition_feature = composition_feature
        self.api_key = api_key

    def fit(self, df, y=None):
        self.original_features = df.columns
        return self

    def transform(self, df):
        # make citrine api call (uses internet)
        cfg = CitrineFeatureGeneration(df.copy(), self.api_key, self.composition_feature)
        df = cfg.generate_citrine_features()

        df = df.drop(self.original_features, axis=1)
        # delete missing values, generation makes a lot of garbage.
        df = clean_dataframe(df)
        assert self.composition_feature not in df.columns
        return df

class PassThrough(BaseEstimator, TransformerMixin):
    def __init__(self, features):
        self.features = features
    def fit(self, df, y=None):
        return self
    def transform(self, df):
        df = df[self.features]
        return df


name_to_constructor = {
    'DoNothing': DoNothing,
    'PolynomialFeatures': PolynomialFeatures,
    'Magpie':Magpie,
    'PassThrough':PassThrough
}

def clean_dataframe(df):
    """ Delete missing values or non-numerics """
    df = df.apply(pd.to_numeric) # convert non-number to NaN

    # drop empty rows
    before_count = df.shape[0]
    df = df.dropna(axis=0, how='all')
    lost_count = before_count - df.shape[0]
    if lost_count > 0:
        warnings.warn(f'Dropping {lost_count}/{before_count} rows for being totally empty')

    # drop columns with any empty cells
    before_count = len(df.columns)
    df = df.select_dtypes(['number']).dropna(axis=1)
    lost_count = before_count - len(df.columns)
    if lost_count > 0:
        warnings.warn(f'Dropping {lost_count}/{before_count} generated columns due to missing values')
    return df


class MagpieFeatureGeneration(object): #TODO update docs once tests are passing

    def __init__(self, dataframe, composition_feature):
        self.dataframe = dataframe
        self.composition_feature = composition_feature

    def generate_magpie_features(self):
        compositions = []
        composition_components = []

        # Replace empty composition fields with empty string instead of NaN
        self.dataframe = self.dataframe.fillna('')
        for column in self.dataframe.columns:
            if self.composition_feature in column:
                composition_components.append(self.dataframe[column].tolist())

        if len(composition_components) < 1:
            print('Error! No column named "Material compositions" found in your input data file. To use this feature generation routine, you must supply a material composition for each data point')
            raise KeyError('No column named "MaterialCompositions" in data file.')

        row = 0
        while row < len(composition_components[0]):
            composition = ''
            for composition_component in composition_components:
                composition += str(composition_component[row])
            compositions.append(composition)
            row += 1

        # Add the column of combined material compositions into the dataframe
        self.dataframe[self.composition_feature] = compositions

        # Assign each magpiedata feature set to appropriate composition name
        magpiedata_dict_composition_average = {}
        magpiedata_dict_arithmetic_average = {}
        magpiedata_dict_max = {}
        magpiedata_dict_min = {}
        magpiedata_dict_difference = {}
        magpiedata_dict_atomic_bysite = {}
        for composition in compositions:
            magpiedata_composition_average, magpiedata_arithmetic_average, magpiedata_max, magpiedata_min, magpiedata_difference = self._get_computed_magpie_features(composition=composition, data_path=MAGPIE_DATA_PATH)
            magpiedata_atomic_notparsed = self._get_atomic_magpie_features(composition=composition, data_path=MAGPIE_DATA_PATH)

            magpiedata_dict_composition_average[composition] = magpiedata_composition_average
            magpiedata_dict_arithmetic_average[composition] = magpiedata_arithmetic_average
            magpiedata_dict_max[composition] = magpiedata_max
            magpiedata_dict_min[composition] = magpiedata_min
            magpiedata_dict_difference[composition] = magpiedata_difference

            # Add site-specific elemental features
            count = 1
            magpiedata_atomic_bysite = {}
            for entry in magpiedata_atomic_notparsed:
                for magpiefeature, featurevalue in magpiedata_atomic_notparsed[entry].items():
                    magpiedata_atomic_bysite["Site"+str(count)+"_"+str(magpiefeature)] = featurevalue
                count += 1

            magpiedata_dict_atomic_bysite[composition] = magpiedata_atomic_bysite

        magpiedata_dict_list = [magpiedata_dict_composition_average, magpiedata_dict_arithmetic_average,
                                magpiedata_dict_max, magpiedata_dict_min, magpiedata_dict_difference, magpiedata_dict_atomic_bysite]

        dataframe = self.dataframe
        for magpiedata_dict in magpiedata_dict_list:
            dataframe_magpie = pd.DataFrame.from_dict(data=magpiedata_dict, orient='index')
            # Need to reorder compositions in new dataframe to match input dataframe
            dataframe_magpie = dataframe_magpie.reindex(self.dataframe[self.composition_feature].tolist())
            # Need to make compositions the first column, instead of the row names
            dataframe_magpie.index.name = self.composition_feature
            dataframe_magpie.reset_index(inplace=True)
            # Need to delete duplicate column before merging dataframes
            del dataframe_magpie[self.composition_feature]
            # Merge magpie feature dataframe with originally supplied dataframe
            dataframe = DataframeUtilities().merge_dataframe_columns(dataframe1=dataframe, dataframe2=dataframe_magpie)

        return dataframe

    def _get_computed_magpie_features(self, composition, data_path):
        magpiedata_composition_average = {}
        magpiedata_arithmetic_average = {}
        magpiedata_max = {}
        magpiedata_min = {}
        magpiedata_difference = {}
        magpiedata_atomic = self._get_atomic_magpie_features(composition=composition, data_path=data_path)
        composition = Composition(composition)
        element_list, atoms_per_formula_unit = self._get_element_list(composition=composition)

        # Initialize feature values to all be 0, because need to dynamically update them with weighted values in next loop.
        for magpie_feature in magpiedata_atomic[element_list[0]]:
            magpiedata_composition_average[magpie_feature] = 0
            magpiedata_arithmetic_average[magpie_feature] = 0
            magpiedata_max[magpie_feature] = 0
            magpiedata_min[magpie_feature] = 0
            magpiedata_difference[magpie_feature] = 0

        for element in magpiedata_atomic:
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
                    # Difference features (max - min)
                    magpiedata_difference[magpie_feature] = magpiedata_max[magpie_feature] - magpiedata_min[magpie_feature]

        # Change names of features to reflect each computed type of magpie feature (max, min, etc.)
        magpiedata_composition_average_renamed = {}
        magpiedata_arithmetic_average_renamed = {}
        magpiedata_max_renamed = {}
        magpiedata_min_renamed = {}
        magpiedata_difference_renamed = {}
        for key in magpiedata_composition_average:
            magpiedata_composition_average_renamed[key+"_composition_average"] = magpiedata_composition_average[key]
        for key in magpiedata_arithmetic_average:
            magpiedata_arithmetic_average_renamed[key+"_arithmetic_average"] = magpiedata_arithmetic_average[key]
        for key in magpiedata_max:
            magpiedata_max_renamed[key+"_max_value"] = magpiedata_max[key]
        for key in magpiedata_min:
            magpiedata_min_renamed[key+"_min_value"] = magpiedata_min[key]
        for key in magpiedata_difference:
            magpiedata_difference_renamed[key+"_difference"] = magpiedata_difference[key]

        return magpiedata_composition_average_renamed, magpiedata_arithmetic_average_renamed, magpiedata_max_renamed, magpiedata_min_renamed, magpiedata_difference_renamed

    def _get_atomic_magpie_features(self, composition, data_path):
        # Get .table files containing feature values for each element, assign file names as feature names
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
            atomic_values = {}
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

    def _get_element_list(self, composition):
        element_amounts = composition.get_el_amt_dict()
        atoms_per_formula_unit = 0
        for v in element_amounts.values():
            atoms_per_formula_unit += v

        # Get list of unique elements present
        element_list = []
        for k in element_amounts:
            if k not in element_list:
                element_list.append(k)

        return element_list, atoms_per_formula_unit


class MaterialsProjectFeatureGeneration(object):
    """
    Class to generate new features using Materials Project data and dataframe containing material compositions
    Datarame must have a column named "Material compositions".

    Args:
        configdict (dict) : MASTML configfile object as dict
        dataframe (pandas dataframe) : dataframe containing x and y data and feature names
        mapi_key (str) : your Materials Project API key

    Methods:
        generate_materialsproject_features : generates materials project feature set based on compositions in dataframe

            Args:
                save_to_csv (bool) : whether to save the magpie feature set to a csv file

            Returns:
                pandas dataframe : dataframe containing magpie feature set
    """
    def __init__(self, dataframe, mapi_key, composition_feature):
        self.dataframe = dataframe
        self.mapi_key = mapi_key
        self.composition_feature = composition_feature

    def generate_materialsproject_features(self):
        try:
            compositions = self.dataframe[self.composition_feature]
        except KeyError as e:
            raise Exception(f'No column named {self.composition_feature} in csv file')

        mpdata_dict_composition = {}

        # before: 11 hits for a total of ~6 seconds
        #for composition in compositions:
        #    composition_data_mp = self._get_data_from_materials_project(composition=composition)
        #    mpdata_dict_composition[composition] = composition_data_mp
        # after: 2.5 seconds!!!
        pool = multiprocessing.Pool(processes=20)
        #comp_data_mp = pool.map(self._get_data_from_materials_project, compositions)
        comp_data_mp = map(self._get_data_from_materials_project, compositions)

        mpdata_dict_composition.update(dict(zip(compositions, comp_data_mp)))

        dataframe = self.dataframe
        dataframe_mp = pd.DataFrame.from_dict(data=mpdata_dict_composition, orient='index')
        # Need to reorder compositions in new dataframe to match input dataframe
        dataframe_mp = dataframe_mp.reindex(self.dataframe[self.composition_feature].tolist())
        # Need to make compositions the first column, instead of the row names
        dataframe_mp.index.name = self.composition_feature
        dataframe_mp.reset_index(inplace=True)
        # Need to delete duplicate column before merging dataframes
        del dataframe_mp[self.composition_feature]
        # Merge magpie feature dataframe with originally supplied dataframe
        dataframe = DataframeUtilities().merge_dataframe_columns(dataframe1=dataframe, dataframe2=dataframe_mp)

        return dataframe

    def _get_data_from_materials_project(self, composition):
        mprester = MPRester(self.mapi_key)
        structure_data_list = mprester.get_data(chemsys_formula_id=composition)

        # Sort structures by stability (i.e. E above hull), and only return most stable compound data
        if len(structure_data_list) > 0:
            structure_data_list = sorted(structure_data_list, key=lambda e_above: e_above['e_above_hull'])
            structure_data_most_stable = structure_data_list[0]
        else:
            structure_data_most_stable = {}

        # Trim down the full Materials Project data dict to include only quantities relevant to make features
        structure_data_dict_condensed = {}
        property_list = ["G_Voigt_Reuss_Hill", "G_Reuss", "K_Voigt_Reuss_Hill", "K_Reuss", "K_Voigt", "G_Voigt", "G_VRH",
                         "homogeneous_poisson", "poisson_ratio", "universal_anisotropy", "K_VRH", "elastic_anisotropy",
                         "band_gap", "e_above_hull", "formation_energy_per_atom", "nelements", "energy_per_atom", "volume",
                         "density", "total_magnetization", "number"]
        elastic_property_list = ["G_Voigt_Reuss_Hill", "G_Reuss", "K_Voigt_Reuss_Hill", "K_Reuss", "K_Voigt", "G_Voigt",
                                 "G_VRH", "homogeneous_poisson", "poisson_ratio", "universal_anisotropy", "K_VRH", "elastic_anisotropy"]
        if len(structure_data_list) > 0:
            for prop in property_list:
                if prop in elastic_property_list:
                    try:
                        structure_data_dict_condensed[prop] = structure_data_most_stable["elasticity"][prop]
                    except TypeError:
                        structure_data_dict_condensed[prop] = ''
                elif prop == "number":
                    try:
                        structure_data_dict_condensed["Spacegroup_"+prop] = structure_data_most_stable["spacegroup"][prop]
                    except TypeError:
                        structure_data_dict_condensed[prop] = ''
                else:
                    try:
                        structure_data_dict_condensed[prop] = structure_data_most_stable[prop]
                    except TypeError:
                        structure_data_dict_condensed[prop] = ''
        else:
            for prop in property_list:
                if prop == "number":
                    structure_data_dict_condensed["Spacegroup_"+prop] = ''
                else:
                    structure_data_dict_condensed[prop] = ''

        if all(val == '' for _, val in structure_data_dict_condensed.items()):
            warnings.warn(f'No data found for composition "{composition}" using materials project')
        else:
            print('MAterials Project Feature Generation', composition, structure_data_dict_condensed)
        return structure_data_dict_condensed


class CitrineFeatureGeneration(object):
    """
    Class to generate new features using Citrine data and dataframe containing material compositions
    Datarame must have a column named "Material compositions".


    Args:
        configdict (dict) : MASTML configfile object as dict
        dataframe (pandas dataframe) : dataframe containing x and y data and feature names
        api_key (str) : your Citrination API key

    Methods:
        generate_citrine_features : generates Citrine feature set based on compositions in dataframe

            Args:
                save_to_csv (bool) : whether to save the magpie feature set to a csv file

            Returns:
                pandas dataframe : dataframe containing magpie feature set
    """
    def __init__(self, dataframe, api_key, composition_feature):
        self.dataframe = dataframe
        self.api_key = api_key
        self.client = CitrinationClient(api_key, 'https://citrination.com')
        self.composition_feature = composition_feature

    def generate_citrine_features(self):
        print('WARNING: You have specified generation of features from Citrine. Based on which'
              ' materials you are interested in, there may be many records to parse through, thus'
              ' this routine may take a long time to complete!')
        try:
            compositions = self.dataframe[self.composition_feature].tolist()
        except KeyError as e:
            print('Error! No column named "Material compositions" found in your input data file. To use this feature generation routine, you must supply a material composition for each data point')
            raise e
        citrine_dict_property_min = dict()
        citrine_dict_property_max = dict()
        citrine_dict_property_avg = dict()

        # before: ~11 seconds
        # made into a func so we can do requests in parallel

        # now like 1.8 secs!
        pool = multiprocessing.Pool(processes=20)
        #result_tuples = pool.map(self._load_composition, compositions)
        result_tuples = map(self._load_composition, compositions)
        for comp, (prop_min, prop_max, prop_avg) in zip(compositions, result_tuples):
            citrine_dict_property_min[comp] = prop_min
            citrine_dict_property_max[comp] = prop_max
            citrine_dict_property_avg[comp] = prop_avg

        dataframe = self.dataframe
        citrine_dict_list = [citrine_dict_property_min, citrine_dict_property_max, citrine_dict_property_avg]
        for citrine_dict in citrine_dict_list:
            dataframe_citrine = pd.DataFrame.from_dict(data=citrine_dict, orient='index')
            # Need to reorder compositions in new dataframe to match input dataframe
            dataframe_citrine = dataframe_citrine.reindex(self.dataframe[self.composition_feature].tolist())
            # Need to make compositions the first column, instead of the row names
            dataframe_citrine.index.name = self.composition_feature
            dataframe_citrine.reset_index(inplace=True)
            # Need to delete duplicate column before merging dataframes
            del dataframe_citrine[self.composition_feature]
            # Merge magpie feature dataframe with originally supplied dataframe
            dataframe = DataframeUtilities().merge_dataframe_columns(dataframe1=dataframe, dataframe2=dataframe_citrine)


        return dataframe

    def _load_composition(self, composition):
        pifquery = self._get_pifquery(composition=composition)
        property_name_list, property_value_list = self._get_pifquery_property_list(pifquery=pifquery)
        print("Citrine Feature Generation: ", composition, property_name_list, property_value_list)
        property_names_unique, parsed_property_min, parsed_property_max, parsed_property_avg = self._parse_pifquery_property_list(property_name_list=property_name_list, property_value_list=property_value_list)
        return parsed_property_min, parsed_property_max, parsed_property_avg


    def _get_pifquery(self, composition):
        # TODO: does this stop csv generation on first invalid composition?
        # TODO: Is there a way to send many compositions in one call to citrine?
        pif_query = PifQuery(system=SystemQuery(chemical_formula=ChemicalFieldQuery(filter=ChemicalFilter(equal=composition))))
        # Check if any results found
        if 'hits' not in self.client.search(pif_query).as_dictionary():
            raise KeyError('No results found!')
        pifquery = self.client.search(pif_query).as_dictionary()['hits']
        return pifquery

    def _get_pifquery_property_list(self, pifquery):
        property_name_list = list()
        property_value_list = list()
        accepted_properties_list = [
            'mass', 'space group', 'band', 'Band', 'energy', 'volume', 'density', 'dielectric',
            'Dielectric', 'Enthalpy', 'Convex', 'Magnetization', 'Elements', 'Modulus', 'Shear',
            "Poisson's", 'Elastic', 'Energy'
        ]

        for result_number, results in enumerate(pifquery):
            for i, dictionary in enumerate(results['system']['properties']):
                if 'name' not in dictionary or dictionary['name'] == "CIF File": continue
                value = dictionary['name']
                for entry in accepted_properties_list:
                    if entry not in value: continue
                    property_name_list.append(value)
                    try:
                        property_value_list.append(
                            float(dictionary['scalars'][0]['value']))
                    except (ValueError, KeyError):
                        property_name_list.pop(-1)
                        continue



        #for result_number, results in enumerate(pifquery):
        #    property_value = results['system']['properties']
        #    for list_index, list_element in enumerate(property_value):
        #        for name, value in property_value[list_index].items():
        #            if name == 'name' and value != "CIF File":
        #                for entry in accepted_properties_list:
        #                    if entry in value:
        #                        property_name_list.append(value)
        #                        try:
        #                            property_value_list.append(
        #                                float(property_value[list_index]['scalars'][0]['value']))
        #                        except (ValueError, KeyError):
        #                            # print('found something to remove', property_value[list_index]['scalars'][0]['value'])
        #                            property_name_list.pop(-1)
        #                            continue






        return property_name_list, property_value_list


    def _parse_pifquery_property_list(self, property_name_list, property_value_list):
        parsed_property_max = dict()
        parsed_property_min = dict()
        parsed_property_avg = dict()
        property_names_unique = list()
        if len(property_name_list) != len(property_value_list):
            print('Error! Length of property name and property value lists are not the same. There must be a bug in the _get_pifquerey_property_list method')
            raise IndexError("property_name_list and property_value_list are not the same size.")
        else:
            # Get unique property names
            for name in property_name_list:
                if name not in property_names_unique:
                    property_names_unique.append(name)
            for unique_name in property_names_unique:
                unique_property = list()
                unique_property_avg = 0
                count = 0
                for i, name in enumerate(property_name_list):
                    # Only include property values whose name are same as those in unique_name list
                    if name == unique_name:
                        count += 1 # count how many instances of the same property occur
                        unique_property_avg += property_value_list[i]
                        unique_property.append(property_value_list[i])
                unique_property_min = min(entry for entry in unique_property)
                unique_property_max = max(entry for entry in unique_property)
                unique_property_avg = unique_property_avg/count
                parsed_property_min[str(unique_name)+"_min"] = unique_property_min
                parsed_property_max[str(unique_name) + "_max"] = unique_property_max
                parsed_property_avg[str(unique_name) + "_avg"] = unique_property_avg

        return property_names_unique, parsed_property_min, parsed_property_max, parsed_property_avg


class DataframeUtilities(object):
    """
    Class of basic utilities for dataframe manipulation, and exchanging between dataframes and numpy arrays
    Attributes:
        None
    Methods:
        _merge_dataframe_columns : merge two dataframes by concatenating the column names (duplicate columns omitted)
            args:
                dataframe1 <pandas dataframe> : a pandas dataframe object
                dataframe2 <pandas dataframe> : a pandas dataframe object
            returns:
                dataframe <pandas dataframe> : merged dataframe
        _merge_dataframe_rows : merge two dataframes by concatenating the row contents (duplicate rows omitted)
            args:
                dataframe1 <pandas dataframe> : a pandas dataframe object
                dataframe2 <pandas dataframe> : a pandas dataframe object
            returns:
                dataframe <pandas dataframe> : merged dataframe
        _get_dataframe_statistics : obtain basic statistics about data contained in the dataframe
            args:
                dataframe <pandas dataframe> : a pandas dataframe object
            returns:
                dataframe_stats <pandas dataframe> : dataframe containing input dataframe statistics
        _dataframe_to_array : transform a pandas dataframe to a numpy array
            args:
                dataframe <pandas dataframe> : a pandas dataframe object
            returns:
                array <numpy array> : a numpy array representation of the inputted dataframe
        _array_to_dataframe : transform a numpy array to a pandas dataframe
            args:
                array <numpy array> : a numpy array object
            returns:
                dataframe <pandas dataframe> : a pandas dataframe representation of the inputted numpy array
        _concatenate_arrays : merge two numpy arrays by concatenating along the columns
            args:
                Xarray <numpy array> : a numpy array object
                yarray <numpy array> : a numpy array object
            returns:
                array <numpy array> : a numpy array merging the two input arrays
        _assign_columns_as_features : adds column names to dataframe based on the x and y feature names
            args:
                dataframe <pandas dataframe> : a pandas dataframe object
                x_features <list> : list containing x feature names
                y_feature <str> : target feature name
            returns:
                dataframe <pandas dataframe> : dataframe containing same data as input, with columns labeled with features
        _save_all_dataframe_statistics : obtain dataframe statistics and save it to a csv file
            args:
                dataframe <pandas dataframe> : a pandas dataframe object
                data_path <str> : file path to save dataframe statistics to
            returns:
                fname <str> : name of file dataframe stats saved to
        _plot_dataframe_histogram : creates a histogram plot of target feature data and saves it to designated save path
            args:
                configdict <dict> : MASTML configfile object as dict
                dataframe <pandas dataframe> : a pandas dataframe object
                y_feature <str> : target feature name
            returns:
                fname <str> : name of file dataframe histogram saved to
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
        # Need configdict to get save path
        #if not configfile_path:
        #    configdict = ConfigFileParser(configfile=sys.argv[1]).get_config_dict(path_to_file=os.getcwd())
            #data_path_name = data_path.split('./')[1]
            #data_path_name = data_path_name.split('.csv')[0]
        #    data_path_name = configdict['General Setup']['target_feature']
        #else:
        #    configdict = ConfigFileParser(configfile=configfile_name).get_config_dict(path_to_file=configfile_path)
        data_path_name = configdict['General Setup']['target_feature'] # TODO
        fname = configdict['General Setup']['save_path'] + "/" + 'input_data_statistics_'+data_path_name+'.csv'
        dataframe_stats.to_csv(fname, index=True)
        return fname

    @classmethod
    def plot_dataframe_histogram(cls, configdict, dataframe, y_feature):
        num_bins = int((dataframe.shape[0])/15)
        if num_bins < 1:
            num_bins = 1
        pyplot.hist(x=dataframe[y_feature], bins=num_bins, edgecolor='k')
        pyplot.title('Histogram of ' + y_feature + ' values')
        pyplot.xlabel(y_feature + ' value')
        pyplot.ylabel('Occurrences in dataset')
        fname = configdict['General Setup']['save_path'] + "/" + 'input_data_histogram.pdf' # TODO
        pyplot.savefig(fname)
        return fname
