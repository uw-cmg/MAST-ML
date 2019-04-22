"""
This module contains a collection of classes for generating input features to fit machine learning models to.
"""

import multiprocessing
import os
import logging

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PolynomialFeatures as SklearnPolynomialFeatures

import pymatgen
from pymatgen import Element, Composition
from pymatgen.ext.matproj import MPRester
#from citrination_client import CitrinationClient, PifQuery, SystemQuery, ChemicalFieldQuery, ChemicalFilter
# trouble? try: `pip install citrination_client=="2.1.0"`

# locate path to directory containing AtomicNumber.table, AtomicRadii.table AtomicVolume.table, etc
# (needs to do it the hard way becuase python -m sets cwd to wherever python is ran from)
import mastml
from mastml import utils
log = logging.getLogger('mastml')
# print('mastml dir: ', mastml.__path__)
MAGPIE_DATA_PATH = os.path.join(mastml.__path__[0], 'magpie')

#from matminer.data_retrieval.retrieve_Citrine import CitrineDataRetrieval
#from matminer.data_retrieval.retrieve_MP import MPDataRetrieval

class PolynomialFeatures(BaseEstimator, TransformerMixin):
    """
    Class to generate polynomial features using scikit-learn's polynomial features method
    More info at: http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html

    Args:

        degree: (int), degree of polynomial features

        interaction_only: (bool), If true, only interaction features are produced: features that are products of at
        most degree distinct input features (so not x[1] ** 2, x[0] * x[2] ** 3, etc.).

        include_bias: (bool),If True (default), then include a bias column, the feature in which all polynomial powers
        are zero (i.e. a column of ones - acts as an intercept term in a linear model).

    Methods:

        fit: conducts fit method of polynomial feature generation

        Args:

            df: (dataframe), dataframe of input X and y data

        transform: generates dataframe containing polynomial features

        Args:

            df: (dataframe), dataframe of input X and y data

        Returns:

            (dataframe), dataframe containing new polynomial features, plus original features present

    """
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
        new_features = self.SPF.get_feature_names(self.features)
        return pd.DataFrame(self.SPF.transform(array), columns=new_features)

class ContainsElement(BaseEstimator, TransformerMixin):
    """
    Class to generate new categorical features (i.e. values of 1 or 0) based on whether an input composition contains a
    certain designated element

    Args:

        composition_feature: (str), string denoting a chemical composition to generate elemental features from

        element: (str), string representing the name of an element

        new_name: (str), the name of the new feature column to be generated

        all_elments: (bool), whether to generate new features for all elements present from all compositions in the dataset.

    Methods:

        fit: pass through, needed to maintain scikit-learn class structure

        Args:

            df: (dataframe), dataframe of input X and y data

        transform: generate new element-specific features

        Args:

            df: (dataframe), dataframe of input X and y data

        Returns:

            df_trans: (dataframe), dataframe with generated element-specific features

    """

    def __init__(self, composition_feature, element, new_name, all_elements=False):
        self.composition_feature = composition_feature
        self.element = element
        self.new_column_name = new_name #f'has_{self.element}'
        self.all_elements = all_elements

    def fit(self, df, y=None):
        return self

    def transform(self, df, y=None):
        compositions = df[self.composition_feature]
        if self.all_elements == False:
            has_element = compositions.apply(self._contains_element)
            df_trans = has_element.to_frame(name=self.new_column_name)
        elif self.all_elements == True:
            df_trans = self._contains_all_elements(compositions=compositions)
        return df_trans

    def _contains_element(self, comp):
        """
        Returns 1 if comp contains that element, and 0 if not.
        Uses ints because sklearn and numpy like number classes better than bools. Could even be
        something crazy like "contains {element}" and "does not contain {element}" if you really
        wanted.
        """
        comp = pymatgen.Composition(comp)
        count = comp[self.element]
        return int(count != 0)

    def _contains_all_elements(self, compositions):
        elements = list()
        df_trans = pd.DataFrame()
        for comp in compositions.values:
            comp = pymatgen.Composition(comp)
            for element in comp.elements:
                if element not in elements:
                    elements.append(element)
        for element in elements:
            self.element = element
            self.new_column_name = "has_"+str(self.element)
            has_element = compositions.apply(self._contains_element)
            df_trans[self.new_column_name] = has_element
        return df_trans

class Magpie(BaseEstimator, TransformerMixin):
    """
    Class that wraps MagpieFeatureGeneration, giving it scikit-learn structure

    Args:

        composition_feature: (str), string denoting a chemical composition to generate elemental features from

    Methods:

        fit: pass through, copies input columns as pre-generated features

        Args:

            df: (dataframe), input dataframe containing X and y data

        transform: generate Magpie features

        Args:

            df: (dataframe), input dataframe containing X and y data

        Returns:

            df: (dataframe), output dataframe containing generated features, original features and y data

    """

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
    """
    Class that wraps MaterialsProjectFeatureGeneration, giving it scikit-learn structure

    Args:

        composition_feature: (str), string denoting a chemical composition to generate elemental features from

        mapi_key: (str), string denoting your Materials Project API key

    Methods:

        fit: pass through, copies input columns as pre-generated features

        Args:

            df: (dataframe), input dataframe containing X and y data

        transform: generate Materials Project features

        Args:

            df: (dataframe), input dataframe containing X and y data

        Returns:

            df: (dataframe), output dataframe containing generated features, original features and y data

    """


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
"""

class MaterialsProject(BaseEstimator, TransformerMixin, MPDataRetrieval):

    Class that wraps MaterialsProjectFeatureGeneration, giving it scikit-learn structure

    Args:

        composition_feature: (str), string denoting a chemical composition to generate elemental features from

        mapi_key: (str), string denoting your Materials Project API key

    Methods:

        fit: pass through, copies input columns as pre-generated features

        Args:

            df: (dataframe), input dataframe containing X and y data

        transform: generate Materials Project features

        Args:

            df: (dataframe), input dataframe containing X and y data

        Returns:

            df: (dataframe), output dataframe containing generated features, original features and y data



    def __init__(self, composition_feature, api_key):
        super(MaterialsProject, self).__init__(api_key=api_key)
        self.composition_feature = composition_feature
        self.api_key = api_key
        self.original_features = None

    def fit(self, df, y=None):
        self.original_features = df.columns
        return self

    def transform(self, df=None):
        # make materials project api call (uses internet)
        #mpg = MaterialsProjectFeatureGeneration(df.copy(), self.api_key, self.composition_feature)
        print('HERE')
        print(df[self.composition_feature])
        print(df[self.composition_feature].tolist())
        materials_asstr = self._list2str(lst=df[self.composition_feature].tolist())
        print(materials_asstr)
        df = self.get_dataframe(criteria=df[self.composition_feature].tolist()[0], properties=["formation_energy_per_atom"])
        print(df)
        #df = self.get_dataframe(criteria=materials_asstr, properties=["formation_energy_per_atom"])
        #df = mpg.generate_materialsproject_features()

        df = df.drop(self.original_features, axis=1)
        # delete missing values, generation makes a lot of garbage.
        df = clean_dataframe(df)
        assert self.composition_feature not in df.columns
        return df

    def _list2str(self, lst):
        s = ''
        for l in lst:
            s += l+' '
        return s
"""

"""


class Citrine(BaseEstimator, TransformerMixin, CitrineDataRetrieval):

    Class that wraps MaterialsProjectFeatureGeneration, giving it scikit-learn structure

    Args:

        composition_feature: (str), string denoting a chemical composition to generate elemental features from

        mapi_key: (str), string denoting your Materials Project API key

    Methods:

        fit: pass through, copies input columns as pre-generated features

        Args:

            df: (dataframe), input dataframe containing X and y data

        transform: generate Materials Project features

        Args:

            df: (dataframe), input dataframe containing X and y data

        Returns:

            df: (dataframe), output dataframe containing generated features, original features and y data



    def __init__(self, composition_feature, api_key):
        super(Citrine, self).__init__(api_key=api_key)
        self.composition_feature = composition_feature
        self.api_key = api_key
        self.original_features = None

    def fit(self, df, y=None):
        self.original_features = df.columns
        return self

    def transform(self, df=None):
        # make materials project api call (uses internet)
        #mpg = MaterialsProjectFeatureGeneration(df.copy(), self.api_key, self.composition_feature)
        materials_asstr = self._list2str(lst=df[self.composition_feature].tolist())
        df_count = 0
        props = self._citrine_property_list()
        for i, material in enumerate(df[self.composition_feature].tolist()):
            try:
                df = self.get_dataframe(criteria={'formula': str(material)}, properties=props)
                df_count += 1
                cols = [col for col in df.columns.tolist() if col in props]
                df = df[cols]
                print('FINAL DF')
                print(df)
            except:
                logging.info('MAST-ML encountered a problem generating features from Citrine for', str(material), 'continuing...')
        print('COMPLETED')
        print(df_count)
        exit()
        #df = self.get_dataframe(criteria=materials_asstr, properties=["formation_energy_per_atom"])
        #df = mpg.generate_materialsproject_features()

        df = df.drop(self.original_features, axis=1)
        # delete missing values, generation makes a lot of garbage.
        df = clean_dataframe(df)
        assert self.composition_feature not in df.columns
        return df

    def _list2str(self, lst):
        s = ''
        for l in lst:
            s += l+' '
        return s

    def _citrine_property_list(self):
        props = ['Energy Above Convex Hull', 'Unit Cell Volume', 'VASP Energy for Structure', 'FirstIonizationEnergy_stoich_avg',
                 'HeatFusion_min', 'GSbandgap_min', 'Enthalpy of Formation','Covalent Radius_Max', 'MeltingT_min',
                 'HeatCapacityMolar_min', 'Bulk Modulus, Voigt', 'HeatCapacityMolar_max', 'Density_Min', 'Bulk Modulus, Reuss',
                 'Atomic Weight_Min', 'Density_Max', 'Density', 'HeatFusion_stoich_avg', 'MeltingT_max', 'Molecular mass',
                 'HeatCapacityMolar_stoich_avg', 'Number of Elements', 'Boiling Temp_Stoich Avg', 'GSbandgap_stoich_avg',
                 'LUMO energy', 'Shear Modulus, Voigt', 'Solubility', 'Electronegativity_Max', 'Bulk Modulus, Voigt-Reuss-Hill',
                 'FirstIonizationEnergy_min', 'HOMO energy', 'Atomic Weight_Max', 'Boiling Temp_Max', 'Density_Stoich Avg',
                 'Electronegativity_Stoich Avg', 'Molar Mass', 'Boiling Temp_Min', 'Electronegativity_Min', 'Shear Modulus, Reuss',
                 'HeatFusion_max', 'Shear Modulus, Voigt-Reuss-Hill', 'Dipole moment', 'Heat of formation', 'Melting Point',
                 'Total Magnetization', 'MeltingT_stoich_avg', 'GSbandgap_max', "Poisson's Ratio", 'Volume', 'Atomic Volume_min',
                 'Atomic Volume_max', 'Covalent Radius_Min', 'FirstIonizationEnergy_max']
        return props
"""

class Citrine(BaseEstimator, TransformerMixin):
    """

    Class that wraps CitrineFeatureGeneration, giving it scikit-learn structure

    Args:

        composition_feature: (str), string denoting a chemical composition to generate elemental features from

        api_key: (str), string denoting your Citrine API key

    Methods:

        fit: pass through, copies input columns as pre-generated features

        Args:

            df: (dataframe), input dataframe containing X and y data

        transform: generate Citrine features

        Args:

            df: (dataframe), input dataframe containing X and y data

        Returns:

            df: (dataframe), output dataframe containing generated features, original features and y data
    """



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


class NoGenerate(BaseEstimator, TransformerMixin):
    """
    Class for having a "null" transform where the output is the same as the input. Needed by MAST-ML as a placeholder if
    certain workflow aspects are not performed.

    Args:

        None

    Methods:

        fit: does nothing, just returns object instance. Needed to maintain same structure as scikit-learn classes

        Args:

            X: (dataframe), dataframe of X features

        transform: passes the input back out, in this case the array of X features

        Args:

            X: (dataframe), dataframe of X features

        Returns:

            (dataframe), dataframe of X features

    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return pd.DataFrame(index=X.index)

name_to_constructor = {
    'DoNothing': NoGenerate,
    'PolynomialFeatures': PolynomialFeatures,
    'Magpie': Magpie,
    'Citrine': Citrine,
    'MaterialsProject': MaterialsProject,
    # including these here for now. May get their own section eventually
    'ContainsElement': ContainsElement,
}

def clean_dataframe(df):
    """
    Method to clean dataframes after feature generation has occurred, to remove columns that have a single missing or
    NaN value, or remove a row that is fully empty

    Args:

        df: (dataframe), a post feature generation dataframe that needs cleaning

    Returns:

        df: (dataframe), the cleaned dataframe

    """
    df = df.apply(pd.to_numeric, errors='coerce') # convert non-number to NaN

    # warn on empty rows
    before_count = df.shape[0]
    df = df.dropna(axis=0, how='all')
    lost_count = before_count - df.shape[0]
    if lost_count > 0:
        log.warning(f'Dropping {lost_count}/{before_count} rows for being totally empty')

    # drop columns with any empty cells
    before_count = df.shape[1]
    df = df.select_dtypes(['number']).dropna(axis=1)
    lost_count = before_count - df.shape[1]
    if lost_count > 0:
        log.warning(f'Dropping {lost_count}/{before_count} generated columns due to missing values')
    return df

class MagpieFeatureGeneration(object):
    """
    Class to generate new features using Magpie data and dataframe containing material compositions

    Args:

        dataframe: (pandas dataframe), dataframe containing x and y data and feature names

        composition_feature: (str), string denoting a chemical composition to generate elemental features from

    Methods:

        generate_magpie_features : generates magpie feature set based on compositions in dataframe

            Args:

                None

            Returns:

                dataframe: (dataframe) : dataframe containing magpie feature set
    """

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
            raise utils.MissingColumnError('Error! No column named "Material compositions" found in your input data file. To use this feature generation routine, you must supply a material composition for each data point')

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
                                try:
                                    atomic_values[feature_name] = float(feature_value.strip())
                                except ValueError:
                                    atomic_values[feature_name] = 'NaN'
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
        dataframe: (dataframe), dataframe containing x and y data and feature names

        mapi_key: (str), string denoting your Materials Project API key

        composition_feature: (str), string denoting a chemical composition to generate elemental features from

    Methods:

        generate_materialsproject_features : generates materials project feature set based on compositions in dataframe

            Args:

                None

            Returns:
                dataframe: (dataframe), dataframe containing materials project feature set
    """
    def __init__(self, dataframe, mapi_key, composition_feature):
        self.dataframe = dataframe
        self.mapi_key = mapi_key
        self.composition_feature = composition_feature

    def generate_materialsproject_features(self):
        try:
            compositions = self.dataframe[self.composition_feature]
        except KeyError as e:
            raise utils.MissingColumnError(f'No column named {self.composition_feature} in csv file')

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
            log.warning(f'No data found for composition "{composition}" using materials project')
        else:
            log.info(f'MAterials Project Feature Generation {composition} {structure_data_dict_condensed}')
        return structure_data_dict_condensed

class CitrineFeatureGeneration(object):
    """
    Class to generate new features using Citrine data and dataframe containing material compositions
    Datarame must have a column named "Material compositions".

    Args:
        dataframe: (dataframe), dataframe containing x and y data and feature names

        api_key: (str), your Citrination API key

        composition_feature: (str), string denoting a chemical composition to generate elemental features from

    Methods:

        generate_citrine_features : generates Citrine feature set based on compositions in dataframe

            Args:

                None

            Returns:

                dataframe: (dataframe), dataframe containing citrine generated feature set
    """
    def __init__(self, dataframe, api_key, composition_feature):
        self.dataframe = dataframe
        self.api_key = api_key
        self.client = CitrinationClient(api_key, 'https://citrination.com')
        self.composition_feature = composition_feature

    def generate_citrine_features(self):
        log.warning('WARNING: You have specified generation of features from Citrine. Based on which'
              ' materials you are interested in, there may be many records to parse through, thus'
              ' this routine may take a long time to complete!')
        try:
            compositions = self.dataframe[self.composition_feature].tolist()
        except KeyError as e:
            log.error(f'original python error: {str(e)}')
            raise utils.MissingColumnError('Error! No column named {self.composition_feature} found in your input data file. '
                    'To use this feature generation routine, you must supply a material composition for each data point')
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
        #print("Citrine Feature Generation: ", composition, property_name_list, property_value_list)
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

    Args:

        None

    Methods:

        merge_dataframe_columns : merge two dataframes by concatenating the column names (duplicate columns omitted)

            Args:

                dataframe1: (dataframe), a pandas dataframe object

                dataframe2: (dataframe), a pandas dataframe object

            Returns:

                dataframe: (dataframe), merged dataframe

        merge_dataframe_rows : merge two dataframes by concatenating the row contents (duplicate rows omitted)

            Args:

                dataframe1: (dataframe), a pandas dataframe object

                dataframe2: (dataframe), a pandas dataframe object

            Returns:

                dataframe: (dataframe), merged dataframe

        get_dataframe_statistics : obtain basic statistics about data contained in the dataframe

            Args:

                dataframe: (dataframe), a pandas dataframe object

            Returns:

                dataframe_stats: (dataframe), dataframe containing input dataframe statistics

        dataframe_to_array : transform a pandas dataframe to a numpy array

            Args:

                dataframe: (dataframe), a pandas dataframe object

            Returns:

                array: (numpy array), a numpy array representation of the inputted dataframe

        array_to_dataframe : transform a numpy array to a pandas dataframe

            Args:

                array: (numpy array), a numpy array

            Returns:

                dataframe: (dataframe), a pandas dataframe representation of the inputted numpy array

        concatenate_arrays : merge two numpy arrays by concatenating along the columns

            Args:

                Xarray: (numpy array), a numpy array object

                yarray: (numpy array), a numpy array object

            Returns:

                array: (numpy array), a numpy array merging the two input arrays

        assign_columns_as_features : adds column names to dataframe based on the x and y feature names

            Args:

                dataframe: (dataframe), a pandas dataframe object

                x_features: (list), list containing x feature names

                y_feature: (str), target feature name

            Returns:

                dataframe: (dataframe), dataframe containing same data as input, with columns labeled with features

        save_all_dataframe_statistics : obtain dataframe statistics and save it to a csv file

            Args:

                dataframe: (dataframe), a pandas dataframe object

                data_path: (str), file path to save dataframe statistics to

            Returns:

                fname: (str), name of file dataframe stats saved to

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