"""
This module contains a collection of classes for generating input features to fit machine learning models to.
"""

import multiprocessing
import os
import logging
import re

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PolynomialFeatures as SklearnPolynomialFeatures

import pymatgen
from pymatgen import Element, Composition
from pymatgen.ext.matproj import MPRester

# matminer class imports
import inspect # used to get a dictionary of classes in a module
from matminer.featurizers import structure as struc
from pymatgen.io.vasp.inputs import Poscar
import mastml
from mastml import utils
from matminer.data_retrieval.retrieve_Citrine import CitrineDataRetrieval
from matminer.data_retrieval.retrieve_MP import MPDataRetrieval
from matminer.data_retrieval.retrieve_MDF import MDFDataRetrieval
from matminer.data_retrieval.retrieve_MPDS import MPDSDataRetrieval
from matminer.data_retrieval.retrieve_AFLOW import AFLOWDataRetrieval

# locate path to directory containing AtomicNumber.table, AtomicRadii.table AtomicVolume.table, etc
# (needs to do it the hard way becuase python -m sets cwd to wherever python is ran from)
import mastml
from mastml import utils
log = logging.getLogger('mastml')

MAGPIE_DATA_PATH = os.path.join(mastml.__path__[0], 'magpie')

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

    def __init__(self, composition_feature, feature_types=None):
        self.composition_feature = composition_feature
        self.feature_types = feature_types
        if self.feature_types is None:
            self.feature_types = ['composition_avg', 'arithmetic_avg', 'max', 'min', 'difference', 'elements']

    def fit(self, df, y=None):
        self.original_features = df.columns
        return self

    def transform(self, df):
        mfg = MagpieFeatureGeneration(df, self.composition_feature, self.feature_types)
        df = mfg.generate_magpie_features()
        df = df.drop(self.original_features, axis=1)
        # delete missing values, generation makes a lot of garbage.
        df = clean_dataframe(df)
        df = df.select_dtypes(['number']).dropna(axis=1)
        assert self.composition_feature not in df.columns
        return df[sorted(df.columns.tolist())]

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

class Matminer(BaseEstimator, TransformerMixin):
    """
    Class to generate structural features from matminer structure module
    Args:
        structural_features: the structure feature(s) the user wants to instantiate and generate
        structure_col: the dataframe column that contains the pymatgen structure object. Matminer needs a pymatgen
        structure object in order to instantiate the structural feature
    Methods:
        fit: pass through, needed to maintain scikit-learn class structure
        Args:
            df: (dataframe), dataframe of input x and y data
        transform: main method that iterates through rows of dataframe to create pymatgen structure objects for
        matminer routines. Iterates through list of structural features from conf file and instantiates each structure;
        drops unused dataframe columns and returns the generated features dataframe
        Args:
            df: (dataframe), dataframe containing the path of file to create pymatgen structure object which is under the structure_col
            column
        Returns:
            (dataframe), the generated features dataframe
    """

    def __init__(self, structural_features, structure_col):  # _instantiate only needs this
        # assuming dataframe is coming in with a column 'Structure' with coords.
        # where do I need to raise errors
        if type(structural_features) is str:
            structural_features = [structural_features]

        structural_features = structural_features  # structural feature is now cast as a list
        self.structural_features = structural_features  # structural feature field of class
        self.structure_col = structure_col

    def fit(self, df, y=None):
        return self

    def transform(self, df, y=None):
        # iterate through dataframe rows
        for i, rows in df.iterrows():
            f = Poscar.from_file(df.at[i, self.structure_col])
            structure = f.structure  # create pymatgen structure object
            df.at[i, self.structure_col] = structure  # replace path with structure object

        # iterate through structural_features list
        for struc_feat in range(len(self.structural_features)):
            # nested loop to check structural_features list item against matminer structures list
            for feature_name in inspect.getmembers(struc, inspect.isclass):
                # if structural feature item is a match
                if feature_name[0] == self.structural_features[struc_feat]:
                    sf = getattr(struc, self.structural_features[struc_feat])()  # instantiates the structure featurizer
                    df = sf.fit_featurize_dataframe(df, self.structure_col)  # fit_featurize_dataframe() works for all
                    # updates dataframe if the structural feature happens to be the GlobalSymmetryFeatures
                    if self.structural_features[struc_feat] == 'GlobalSymmetryFeatures':
                        df = df.drop('crystal_system', axis=1)
                        df['is_centrosymmetric'].replace(True, 1, inplace=True)
                        df['is_centrosymmetric'].replace(False, 0, inplace=True)

                    break  # structure feature was found for this iteration, repeat

        # drop unused dataframe columns for rest of application
        df = df.drop(self.structure_col, axis=1)
        df = df.drop('Material', axis=1)
        return df  # return generated dataframe

    def retrieve_mp(self, criteria, properties=["band_gap", "volume", "density", "formation_energy_per_atom"],
                    index_mpid=True, api_key=None):
        """
        Gets data from MP in a dataframe format. See api_link for more details.
        Args:
            criteria (dict): (str/dict) see MPRester.query() for a description of this
                    parameter. String examples: "mp-1234", "Fe2O3", "Li-Fe-O',
                    "\\*2O3". Dict example: {"band_gap": {"$gt": 1}}
            properties ([str]): (list) see MPRester.query() for a description of this
                    parameter. Example: ["formula", "formation_energy_per_atom"]
            plus: "structure", "initial_structure", "final_structure",
                  "bandstructure" (line mode), "bandstructure_uniform",
                  "phonon_bandstructure", "phonon_ddb", "phonon_bandstructure",
                  "phonon_dos". Note that for a long list of compounds, it may
                   take a long time to retrieve some of these objects.
            index_mpid (bool): (bool) Whether to set the materials_id as the dataframe
                    index.
            api_key: (str) Your Materials Project API key, or None if you've
                set up your pymatgen config.
        Returns (pandas.Dataframe): containing results
        notes/bugs: works pretty great, API easy to use and accurate. What to fix for
                    dataframe integration into mastml?
        """
        mp_df = MPDataRetrieval(api_key).get_dataframe(criteria, properties, index_mpid)
        mp_df = mp_df.loc[mp_df['formation_energy_per_atom'].idxmin(), :].to_frame().transpose().reset_index().drop(
            'index', axis=1)

        return mp_df

    def retrieve_citrine(self, criteria, properties, common_fields, secondary_fields, print_properties_options,
                         api_key):
        """
        Gets a Pandas dataframe object from data retrieved from
        the Citrine API.
        Args:
            criteria (dict): see get_data method for supported keys except
                    prop; prop should be included in properties.
            properties ([str]): requested properties/fields/columns.
                    For example, ["Seebeck coefficient", "Band gap"]. If unsure
                    about the exact words, capitalization, etc try something like
                    ["gap"] and "max_results": 3 and print_properties_options=True
                    to see the exact options for this field
            common_fields ([str]): fields that are common to all the requested
                    properties. Common example can be "chemicalFormula". Look for
                    suggested common fields after a quick query for more info
            secondary_fields (bool): if True, fields not included in properties
                    may be added to the output (e.g. references). Recommended only
                    if len(properties)==1'
            print_properties_options (bool): whether to print available options
                    for "properties" and "common_fields" arguments.
            api_key: (str) Your Citrine API key, or None if
                    you've set the CITRINE_KEY environment variable
        return: (object) Pandas dataframe object containing the results
        notes/bugs: criteria needs a dictionary, not specified in get_data() as mentioned,
                    and example on documentation webpage does not work. What to fix for
                    dataframe integration into mastml?
        """

        citrine_df = CitrineDataRetrieval(api_key).get_dataframe(criteria, properties, common_fields, secondary_fields,
                                                                 print_properties_options)
        return citrine_df

    def retrieve_MDF(self, criteria, anonymous=False, properties=None, unwind_arrays=True):
        mdf_df = MDFDataRetrieval(anonymous).get_dataframe(criteria, properties, unwind_arrays)
        return mdf_df

    def retrieve_MPDS(self, criteria, properties=None, api_key=None, endpoint=None):
        mpds_df = MPDSDataRetrieval(api_key, endpoint).get_dataframe(criteria, properties)
        return mpds_df

    def retrieve_AFLOW(self, criteria, properties, files=None, request_size=10000, request_limit=0, index_auid=True):
        aflow_df = AFLOWDataRetrieval().get_dataframe(criteria, properties, files, request_size, index_auid)
        return aflow_df

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
    'Matminer': Matminer,
    'MaterialsProject': MaterialsProject,
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

        feature_types: (list), list containing types of magpie features to include in the final dataframe. Options
        include ["composition_avg", "arithmetic_avg", "max", "min", "difference", "elements"]. Specifying nothing will
        include all features.

    Methods:

        generate_magpie_features : generates magpie feature set based on compositions in dataframe

            Args:

                None

            Returns:

                dataframe: (dataframe) : dataframe containing magpie feature set
    """

    def __init__(self, dataframe, composition_feature, feature_types):
        self.dataframe = dataframe
        self.composition_feature = composition_feature
        self.feature_types = feature_types

    def generate_magpie_features(self):
        # Replace empty composition fields with empty string instead of NaN
        self.dataframe = self.dataframe.fillna('')

        compositions_raw = self.dataframe[self.composition_feature].tolist()
        # Check first entry of comps to find [] for delimiting different sublattices
        has_sublattices = False
        if '[' in compositions_raw[0]:
            if ']' in compositions_raw[0]:
                has_sublattices = True
                log.info('MAGPIE feature generation found brackets in material compositions denoting specific sublattices!')
                # Parse raw composition strings with brackets to denote compositions of different sublattices
                site_dict_list = list()
                for comp in compositions_raw:
                    sites = re.findall(r"\[([A-Za-z0-9_.]+)\]", comp)
                    site_dict = dict()
                    for i, site in enumerate(sites):
                        comp_by_site = Composition(site).as_dict()
                        site_dict['Site'+str(i+1)] = comp_by_site
                    site_dict_list.append(site_dict)

        compositions = list()
        if has_sublattices == True:
            # Parse out brackets from compositions
            for comp in compositions_raw:
                comp_split = comp.split('[')
                comp_str = ''
                for s in comp_split:
                    comp_str += s
                comp_split = comp_str.split(']')
                comp_str = ''
                for s in comp_split:
                    comp_str += s
                compositions.append(comp_str)
        else:
            compositions = compositions_raw

        if len(compositions) < 1:
            raise utils.MissingColumnError('Error! No material compositions column found in your input data file. To use this feature generation routine, you must supply a material composition for each data point')

        # Add the column of combined material compositions into the dataframe
        self.dataframe[self.composition_feature] = compositions

        # Assign each magpiedata feature set to appropriate composition name
        magpiedata_dict_composition_average = {}
        magpiedata_dict_arithmetic_average = {}
        magpiedata_dict_max = {}
        magpiedata_dict_min = {}
        magpiedata_dict_difference = {}

        magpiedata_dict_atomic_bysite = {}

        magpiedata_dict_composition_average_site1 = {}
        magpiedata_dict_arithmetic_average_site1 = {}
        magpiedata_dict_max_site1 = {}
        magpiedata_dict_min_site1 = {}
        magpiedata_dict_difference_site1 = {}

        magpiedata_dict_composition_average_site2 = {}
        magpiedata_dict_arithmetic_average_site2 = {}
        magpiedata_dict_max_site2 = {}
        magpiedata_dict_min_site2 = {}
        magpiedata_dict_difference_site2 = {}

        magpiedata_dict_composition_average_site3 = {}
        magpiedata_dict_arithmetic_average_site3 = {}
        magpiedata_dict_max_site3 = {}
        magpiedata_dict_min_site3 = {}
        magpiedata_dict_difference_site3 = {}

        magpiedata_dict_composition_average_site1site2 = {}
        magpiedata_dict_arithmetic_average_site1site2 = {}
        magpiedata_dict_max_site1site2 = {}
        magpiedata_dict_min_site1site2 = {}
        magpiedata_dict_difference_site1site2 = {}

        magpiedata_dict_composition_average_site1site3 = {}
        magpiedata_dict_arithmetic_average_site1site3 = {}
        magpiedata_dict_max_site1site3 = {}
        magpiedata_dict_min_site1site3 = {}
        magpiedata_dict_difference_site1site3 = {}

        magpiedata_dict_composition_average_site2site3 = {}
        magpiedata_dict_arithmetic_average_site2site3 = {}
        magpiedata_dict_max_site2site3 = {}
        magpiedata_dict_min_site2site3 = {}
        magpiedata_dict_difference_site2site3 = {}

        for i, composition in enumerate(compositions):
            if has_sublattices:
                magpiedata_collected = self._get_computed_magpie_features(composition=composition, data_path=MAGPIE_DATA_PATH, site_dict=site_dict_list[i])
            else:
                magpiedata_collected = self._get_computed_magpie_features(composition=composition,data_path=MAGPIE_DATA_PATH, site_dict=None)
            magpiedata_atomic_notparsed = self._get_atomic_magpie_features(composition=composition, data_path=MAGPIE_DATA_PATH)

            if has_sublattices:
                number_sites = len(site_dict_list[i].keys())
                if number_sites == 1:
                    magpiedata_composition_average = magpiedata_collected[0]
                    magpiedata_arithmetic_average = magpiedata_collected[1]
                    magpiedata_max = magpiedata_collected[2]
                    magpiedata_min = magpiedata_collected[3]
                    magpiedata_difference = magpiedata_collected[4]
                    magpiedata_composition_average_site1 = magpiedata_collected[5]
                    magpiedata_arithmetic_average_site1 = magpiedata_collected[6]
                    magpiedata_max_site1 = magpiedata_collected[7]
                    magpiedata_min_site1 = magpiedata_collected[8]
                    magpiedata_difference_site1 = magpiedata_collected[9]

                    magpiedata_dict_composition_average[composition] = magpiedata_composition_average
                    magpiedata_dict_arithmetic_average[composition] = magpiedata_arithmetic_average
                    magpiedata_dict_max[composition] = magpiedata_max
                    magpiedata_dict_min[composition] = magpiedata_min
                    magpiedata_dict_difference[composition] = magpiedata_difference

                    magpiedata_dict_composition_average_site1[composition] = magpiedata_composition_average_site1
                    magpiedata_dict_arithmetic_average_site1[composition] = magpiedata_arithmetic_average_site1
                    magpiedata_dict_max_site1[composition] = magpiedata_max_site1
                    magpiedata_dict_min_site1[composition] = magpiedata_min_site1
                    magpiedata_dict_difference_site1[composition] = magpiedata_difference_site1

                elif number_sites == 2:
                    magpiedata_composition_average = magpiedata_collected[0]
                    magpiedata_arithmetic_average = magpiedata_collected[1]
                    magpiedata_max = magpiedata_collected[2]
                    magpiedata_min = magpiedata_collected[3]
                    magpiedata_difference = magpiedata_collected[4]
                    magpiedata_composition_average_site1 = magpiedata_collected[5]
                    magpiedata_arithmetic_average_site1 = magpiedata_collected[6]
                    magpiedata_max_site1 = magpiedata_collected[7]
                    magpiedata_min_site1 = magpiedata_collected[8]
                    magpiedata_difference_site1 = magpiedata_collected[9]
                    magpiedata_composition_average_site2 = magpiedata_collected[10]
                    magpiedata_arithmetic_average_site2 = magpiedata_collected[11]
                    magpiedata_max_site2 = magpiedata_collected[12]
                    magpiedata_min_site2 = magpiedata_collected[13]
                    magpiedata_difference_site2 = magpiedata_collected[14]

                    magpiedata_dict_composition_average[composition] = magpiedata_composition_average
                    magpiedata_dict_arithmetic_average[composition] = magpiedata_arithmetic_average
                    magpiedata_dict_max[composition] = magpiedata_max
                    magpiedata_dict_min[composition] = magpiedata_min
                    magpiedata_dict_difference[composition] = magpiedata_difference

                    magpiedata_dict_composition_average_site1[composition] = magpiedata_composition_average_site1
                    magpiedata_dict_arithmetic_average_site1[composition] = magpiedata_arithmetic_average_site1
                    magpiedata_dict_max_site1[composition] = magpiedata_max_site1
                    magpiedata_dict_min_site1[composition] = magpiedata_min_site1
                    magpiedata_dict_difference_site1[composition] = magpiedata_difference_site1

                    magpiedata_dict_composition_average_site2[composition] = magpiedata_composition_average_site2
                    magpiedata_dict_arithmetic_average_site2[composition] = magpiedata_arithmetic_average_site2
                    magpiedata_dict_max_site2[composition] = magpiedata_max_site2
                    magpiedata_dict_min_site2[composition] = magpiedata_min_site2
                    magpiedata_dict_difference_site2[composition] = magpiedata_difference_site2

                elif number_sites == 3:
                    magpiedata_composition_average = magpiedata_collected[0]
                    magpiedata_arithmetic_average = magpiedata_collected[1]
                    magpiedata_max = magpiedata_collected[2]
                    magpiedata_min = magpiedata_collected[3]
                    magpiedata_difference = magpiedata_collected[4]
                    magpiedata_composition_average_site1 = magpiedata_collected[5]
                    magpiedata_arithmetic_average_site1 = magpiedata_collected[6]
                    magpiedata_max_site1 = magpiedata_collected[7]
                    magpiedata_min_site1 = magpiedata_collected[8]
                    magpiedata_difference_site1 = magpiedata_collected[9]
                    magpiedata_composition_average_site2 = magpiedata_collected[10]
                    magpiedata_arithmetic_average_site2 = magpiedata_collected[11]
                    magpiedata_max_site2 = magpiedata_collected[12]
                    magpiedata_min_site2 = magpiedata_collected[13]
                    magpiedata_difference_site2 = magpiedata_collected[14]
                    magpiedata_composition_average_site3 = magpiedata_collected[15]
                    magpiedata_arithmetic_average_site3 = magpiedata_collected[16]
                    magpiedata_max_site3 = magpiedata_collected[17]
                    magpiedata_min_site3 = magpiedata_collected[18]
                    magpiedata_difference_site3 = magpiedata_collected[19]

                    # Couplings between sites
                    magpiedata_composition_average_site1site2 = magpiedata_collected[20]
                    magpiedata_arithmetic_average_site1site2 = magpiedata_collected[21]
                    magpiedata_difference_site1site2 = magpiedata_collected[22]
                    magpiedata_composition_average_site1site3 = magpiedata_collected[23]
                    magpiedata_arithmetic_average_site1site3 = magpiedata_collected[24]
                    magpiedata_difference_site1site3 = magpiedata_collected[25]
                    magpiedata_composition_average_site2site3 = magpiedata_collected[26]
                    magpiedata_arithmetic_average_site2site3 = magpiedata_collected[27]
                    magpiedata_difference_site2site3 = magpiedata_collected[28]

                    magpiedata_dict_composition_average[composition] = magpiedata_composition_average
                    magpiedata_dict_arithmetic_average[composition] = magpiedata_arithmetic_average
                    magpiedata_dict_max[composition] = magpiedata_max
                    magpiedata_dict_min[composition] = magpiedata_min
                    magpiedata_dict_difference[composition] = magpiedata_difference

                    magpiedata_dict_composition_average_site1[composition] = magpiedata_composition_average_site1
                    magpiedata_dict_arithmetic_average_site1[composition] = magpiedata_arithmetic_average_site1
                    magpiedata_dict_max_site1[composition] = magpiedata_max_site1
                    magpiedata_dict_min_site1[composition] = magpiedata_min_site1
                    magpiedata_dict_difference_site1[composition] = magpiedata_difference_site1

                    magpiedata_dict_composition_average_site2[composition] = magpiedata_composition_average_site2
                    magpiedata_dict_arithmetic_average_site2[composition] = magpiedata_arithmetic_average_site2
                    magpiedata_dict_max_site2[composition] = magpiedata_max_site2
                    magpiedata_dict_min_site2[composition] = magpiedata_min_site2
                    magpiedata_dict_difference_site2[composition] = magpiedata_difference_site2

                    magpiedata_dict_composition_average_site3[composition] = magpiedata_composition_average_site3
                    magpiedata_dict_arithmetic_average_site3[composition] = magpiedata_arithmetic_average_site3
                    magpiedata_dict_max_site3[composition] = magpiedata_max_site3
                    magpiedata_dict_min_site3[composition] = magpiedata_min_site3
                    magpiedata_dict_difference_site3[composition] = magpiedata_difference_site3

                    # Site1+Site2 coupling
                    magpiedata_dict_composition_average_site1site2[composition] = magpiedata_composition_average_site1site2
                    magpiedata_dict_arithmetic_average_site1site2[composition] = magpiedata_arithmetic_average_site1site2
                    magpiedata_dict_difference_site1site2[composition] = magpiedata_difference_site1site2

                    # Site1+Site3 coupling
                    magpiedata_dict_composition_average_site1site3[composition] = magpiedata_composition_average_site1site3
                    magpiedata_dict_arithmetic_average_site1site3[composition] = magpiedata_arithmetic_average_site1site3
                    magpiedata_dict_difference_site1site3[composition] = magpiedata_difference_site1site3

                    # Site2+Site3 coupling
                    magpiedata_dict_composition_average_site2site3[composition] = magpiedata_composition_average_site2site3
                    magpiedata_dict_arithmetic_average_site2site3[composition] = magpiedata_arithmetic_average_site2site3
                    magpiedata_dict_difference_site2site3[composition] = magpiedata_difference_site2site3

            else:
                magpiedata_composition_average = magpiedata_collected[0]
                magpiedata_arithmetic_average = magpiedata_collected[1]
                magpiedata_max = magpiedata_collected[2]
                magpiedata_min = magpiedata_collected[3]
                magpiedata_difference = magpiedata_collected[4]

                magpiedata_dict_composition_average[composition] = magpiedata_composition_average
                magpiedata_dict_arithmetic_average[composition] = magpiedata_arithmetic_average
                magpiedata_dict_max[composition] = magpiedata_max
                magpiedata_dict_min[composition] = magpiedata_min
                magpiedata_dict_difference[composition] = magpiedata_difference

            count = 1
            magpiedata_atomic_bysite = {}
            # Also include magpie features of individual elements in the material
            for entry in magpiedata_atomic_notparsed:
                for magpiefeature, featurevalue in magpiedata_atomic_notparsed[entry].items():
                    magpiedata_atomic_bysite["Element"+str(count)+"_"+str(magpiefeature)] = featurevalue
                count += 1

            magpiedata_dict_atomic_bysite[composition] = magpiedata_atomic_bysite

        if has_sublattices:
            if number_sites == 1:
                magpiedata_dict_list = [magpiedata_dict_composition_average, magpiedata_dict_arithmetic_average,
                                    magpiedata_dict_max, magpiedata_dict_min, magpiedata_dict_difference, magpiedata_dict_atomic_bysite,
                                        magpiedata_dict_composition_average_site1, magpiedata_dict_arithmetic_average_site1,
                                        magpiedata_dict_max_site1, magpiedata_dict_min_site1, magpiedata_dict_difference_site1]
            elif number_sites == 2:
                magpiedata_dict_list = [magpiedata_dict_composition_average, magpiedata_dict_arithmetic_average,
                                    magpiedata_dict_max, magpiedata_dict_min, magpiedata_dict_difference, magpiedata_dict_atomic_bysite,
                                        magpiedata_dict_composition_average_site1, magpiedata_dict_arithmetic_average_site1,
                                        magpiedata_dict_max_site1, magpiedata_dict_min_site1, magpiedata_dict_difference_site1,
                                        magpiedata_dict_composition_average_site2, magpiedata_dict_arithmetic_average_site2,
                                        magpiedata_dict_max_site2, magpiedata_dict_min_site2, magpiedata_dict_difference_site2]
            elif number_sites == 3:
                magpiedata_dict_list = [magpiedata_dict_composition_average, magpiedata_dict_arithmetic_average,
                                    magpiedata_dict_max, magpiedata_dict_min, magpiedata_dict_difference, magpiedata_dict_atomic_bysite,
                                        magpiedata_dict_composition_average_site1, magpiedata_dict_arithmetic_average_site1,
                                        magpiedata_dict_max_site1, magpiedata_dict_min_site1, magpiedata_dict_difference_site1,
                                        magpiedata_dict_composition_average_site2, magpiedata_dict_arithmetic_average_site2,
                                        magpiedata_dict_max_site2, magpiedata_dict_min_site2, magpiedata_dict_difference_site2,
                                        magpiedata_dict_composition_average_site3, magpiedata_dict_arithmetic_average_site3,
                                        magpiedata_dict_max_site3, magpiedata_dict_min_site3, magpiedata_dict_difference_site3,
                                        magpiedata_dict_composition_average_site1site2, magpiedata_dict_arithmetic_average_site1site2, magpiedata_dict_difference_site1site2,
                                        magpiedata_dict_composition_average_site1site3, magpiedata_dict_arithmetic_average_site1site3, magpiedata_dict_difference_site1site3,
                                        magpiedata_dict_composition_average_site2site3, magpiedata_dict_arithmetic_average_site2site3, magpiedata_dict_difference_site2site3]
        else:
            magpiedata_dict_list = [magpiedata_dict_composition_average, magpiedata_dict_arithmetic_average,
                                magpiedata_dict_max, magpiedata_dict_min, magpiedata_dict_difference, magpiedata_dict_atomic_bysite]

        dataframe = self.dataframe
        magpiedata_dict_list_toinclude = list()
        if 'composition_avg' in self.feature_types:
            magpiedata_dict_list_toinclude.append(magpiedata_dict_list[0])
        if 'arithmetic_avg' in self.feature_types:
            magpiedata_dict_list_toinclude.append(magpiedata_dict_list[1])
        if 'max' in self.feature_types:
            magpiedata_dict_list_toinclude.append(magpiedata_dict_list[2])
        if 'min' in self.feature_types:
            magpiedata_dict_list_toinclude.append(magpiedata_dict_list[3])
        if 'difference' in self.feature_types:
            magpiedata_dict_list_toinclude.append(magpiedata_dict_list[4])
        if 'elements' in self.feature_types:
            magpiedata_dict_list_toinclude.append(magpiedata_dict_list[5])
        if has_sublattices is True:
            if number_sites == 1:
                if 'composition_avg' in self.feature_types:
                    magpiedata_dict_list_toinclude.append(magpiedata_dict_list[6])
                if 'arithmetic_avg' in self.feature_types:
                    magpiedata_dict_list_toinclude.append(magpiedata_dict_list[7])
                if 'max' in self.feature_types:
                    magpiedata_dict_list_toinclude.append(magpiedata_dict_list[8])
                if 'min' in self.feature_types:
                    magpiedata_dict_list_toinclude.append(magpiedata_dict_list[9])
                if 'difference' in self.feature_types:
                    magpiedata_dict_list_toinclude.append(magpiedata_dict_list[10])
            if number_sites == 2:
                if 'composition_avg' in self.feature_types:
                    magpiedata_dict_list_toinclude.append(magpiedata_dict_list[6])
                    magpiedata_dict_list_toinclude.append(magpiedata_dict_list[11])
                if 'arithmetic_avg' in self.feature_types:
                    magpiedata_dict_list_toinclude.append(magpiedata_dict_list[7])
                    magpiedata_dict_list_toinclude.append(magpiedata_dict_list[12])
                if 'max' in self.feature_types:
                    magpiedata_dict_list_toinclude.append(magpiedata_dict_list[8])
                    magpiedata_dict_list_toinclude.append(magpiedata_dict_list[13])
                if 'min' in self.feature_types:
                    magpiedata_dict_list_toinclude.append(magpiedata_dict_list[9])
                    magpiedata_dict_list_toinclude.append(magpiedata_dict_list[14])
                if 'difference' in self.feature_types:
                    magpiedata_dict_list_toinclude.append(magpiedata_dict_list[10])
                    magpiedata_dict_list_toinclude.append(magpiedata_dict_list[15])
            if number_sites == 3:
                if 'composition_avg' in self.feature_types:
                    magpiedata_dict_list_toinclude.append(magpiedata_dict_list[6])
                    magpiedata_dict_list_toinclude.append(magpiedata_dict_list[11])
                    magpiedata_dict_list_toinclude.append(magpiedata_dict_list[16])
                    if 'Site1Site2' in self.feature_types:
                        magpiedata_dict_list_toinclude.append(magpiedata_dict_list[21])
                    if 'Site1Site3' in self.feature_types:
                        magpiedata_dict_list_toinclude.append(magpiedata_dict_list[24])
                    if 'Site2Site3' in self.feature_types:
                        magpiedata_dict_list_toinclude.append(magpiedata_dict_list[27])
                if 'arithmetic_avg' in self.feature_types:
                    magpiedata_dict_list_toinclude.append(magpiedata_dict_list[7])
                    magpiedata_dict_list_toinclude.append(magpiedata_dict_list[12])
                    magpiedata_dict_list_toinclude.append(magpiedata_dict_list[17])
                    if 'Site1Site2' in self.feature_types:
                        magpiedata_dict_list_toinclude.append(magpiedata_dict_list[22])
                    if 'Site1Site3' in self.feature_types:
                        magpiedata_dict_list_toinclude.append(magpiedata_dict_list[25])
                    if 'Site2Site3' in self.feature_types:
                        magpiedata_dict_list_toinclude.append(magpiedata_dict_list[28])
                if 'max' in self.feature_types:
                    magpiedata_dict_list_toinclude.append(magpiedata_dict_list[8])
                    magpiedata_dict_list_toinclude.append(magpiedata_dict_list[13])
                    magpiedata_dict_list_toinclude.append(magpiedata_dict_list[18])
                if 'min' in self.feature_types:
                    magpiedata_dict_list_toinclude.append(magpiedata_dict_list[9])
                    magpiedata_dict_list_toinclude.append(magpiedata_dict_list[14])
                    magpiedata_dict_list_toinclude.append(magpiedata_dict_list[19])
                if 'difference' in self.feature_types:
                    magpiedata_dict_list_toinclude.append(magpiedata_dict_list[10])
                    magpiedata_dict_list_toinclude.append(magpiedata_dict_list[15])
                    magpiedata_dict_list_toinclude.append(magpiedata_dict_list[20])
                    if 'Site1Site2' in self.feature_types:
                        magpiedata_dict_list_toinclude.append(magpiedata_dict_list[23])
                    if 'Site1Site3' in self.feature_types:
                        magpiedata_dict_list_toinclude.append(magpiedata_dict_list[26])
                    if 'Site2Site3' in self.feature_types:
                        magpiedata_dict_list_toinclude.append(magpiedata_dict_list[29])

        for magpiedata_dict in magpiedata_dict_list_toinclude:
            dataframe_magpie = pd.DataFrame.from_dict(data=magpiedata_dict, orient='index')
            # Need to reorder compositions in new dataframe to match input dataframe
            dataframe_magpie = dataframe_magpie.reindex(self.dataframe[self.composition_feature].tolist())
            # Need to make compositions the first column, instead of the row names
            dataframe_magpie.index.name = self.composition_feature
            dataframe_magpie.reset_index(inplace=True)
            # Merge magpie feature dataframe with originally supplied dataframe
            dataframe = DataframeUtilities().merge_dataframe_columns(dataframe1=dataframe, dataframe2=dataframe_magpie)

        return dataframe

    def _get_computed_magpie_features(self, composition, data_path, site_dict=None):
        magpiedata_composition_average = {}
        magpiedata_arithmetic_average = {}
        magpiedata_max = {}
        magpiedata_min = {}
        magpiedata_difference = {}
        magpiedata_atomic = self._get_atomic_magpie_features(composition=composition, data_path=data_path)
        composition = Composition(composition)
        element_list, atoms_per_formula_unit = self._get_element_list(composition=composition)

        # Make per-site dicts if site_dict_list specified
        if site_dict:
            number_sites = len(site_dict.keys())
            for site, comp_dict in site_dict.items():
                if site == "Site1":
                    num_site1_elements = int(len(site_dict[site].keys()))
                    site1_total = 0
                    for el, amt in comp_dict.items():
                        site1_total += amt
                if site == "Site2":
                    num_site2_elements = int(len(site_dict[site].keys()))
                    site2_total = 0
                    for el, amt in comp_dict.items():
                        site2_total += amt
                if site == "Site3":
                    num_site3_elements = int(len(site_dict[site].keys()))
                    site3_total = 0
                    for el, amt in comp_dict.items():
                        site3_total += amt
            if number_sites == 1:
                magpiedata_composition_average_site1 = {}
                magpiedata_arithmetic_average_site1 = {}
                magpiedata_max_site1 = {}
                magpiedata_min_site1 = {}
                magpiedata_difference_site1 = {}
            elif number_sites == 2:
                magpiedata_composition_average_site1 = {}
                magpiedata_arithmetic_average_site1 = {}
                magpiedata_max_site1 = {}
                magpiedata_min_site1 = {}
                magpiedata_difference_site1 = {}
                magpiedata_composition_average_site2 = {}
                magpiedata_arithmetic_average_site2 = {}
                magpiedata_max_site2 = {}
                magpiedata_min_site2 = {}
                magpiedata_difference_site2 = {}
            elif number_sites == 3:
                magpiedata_composition_average_site1 = {}
                magpiedata_arithmetic_average_site1 = {}
                magpiedata_max_site1 = {}
                magpiedata_min_site1 = {}
                magpiedata_difference_site1 = {}
                magpiedata_composition_average_site2 = {}
                magpiedata_arithmetic_average_site2 = {}
                magpiedata_max_site2 = {}
                magpiedata_min_site2 = {}
                magpiedata_difference_site2 = {}
                magpiedata_composition_average_site3 = {}
                magpiedata_arithmetic_average_site3 = {}
                magpiedata_max_site3 = {}
                magpiedata_min_site3 = {}
                magpiedata_difference_site3 = {}
                # Couplings between sites
                magpiedata_composition_average_site1site2 = {}
                magpiedata_arithmetic_average_site1site2 = {}
                magpiedata_difference_site1site2 = {}
                magpiedata_composition_average_site1site3 = {}
                magpiedata_arithmetic_average_site1site3 = {}
                magpiedata_difference_site1site3 = {}
                magpiedata_composition_average_site2site3 = {}
                magpiedata_arithmetic_average_site2site3 = {}
                magpiedata_difference_site2site3 = {}

            else:
                log.error('MASTML currently only supports up to 3 sublattices to generate site-specific MAGPIE features. '
                          'Please reduce number of sublattices an re-run MASTML.')

        # Initialize feature values to all be 0, because need to dynamically update them with weighted values in next loop.
        for magpie_feature in magpiedata_atomic[element_list[0]]:
            magpiedata_composition_average[magpie_feature] = 0
            magpiedata_arithmetic_average[magpie_feature] = 0
            magpiedata_max[magpie_feature] = 0
            magpiedata_min[magpie_feature] = 0
            magpiedata_difference[magpie_feature] = 0

            if site_dict:
                if number_sites == 1:
                    magpiedata_composition_average_site1[magpie_feature] = 0
                    magpiedata_arithmetic_average_site1[magpie_feature] = 0
                    magpiedata_max_site1[magpie_feature] = 0
                    magpiedata_min_site1[magpie_feature] = 0
                    magpiedata_difference_site1[magpie_feature] = 0
                elif number_sites == 2:
                    magpiedata_composition_average_site1[magpie_feature] = 0
                    magpiedata_arithmetic_average_site1[magpie_feature] = 0
                    magpiedata_max_site1[magpie_feature] = 0
                    magpiedata_min_site1[magpie_feature] = 0
                    magpiedata_difference_site1[magpie_feature] = 0
                    magpiedata_composition_average_site2[magpie_feature] = 0
                    magpiedata_arithmetic_average_site2[magpie_feature] = 0
                    magpiedata_max_site2[magpie_feature] = 0
                    magpiedata_min_site2[magpie_feature] = 0
                    magpiedata_difference_site2[magpie_feature] = 0
                elif number_sites == 3:
                    magpiedata_composition_average_site1[magpie_feature] = 0
                    magpiedata_arithmetic_average_site1[magpie_feature] = 0
                    magpiedata_max_site1[magpie_feature] = 0
                    magpiedata_min_site1[magpie_feature] = 0
                    magpiedata_difference_site1[magpie_feature] = 0
                    magpiedata_composition_average_site2[magpie_feature] = 0
                    magpiedata_arithmetic_average_site2[magpie_feature] = 0
                    magpiedata_max_site2[magpie_feature] = 0
                    magpiedata_min_site2[magpie_feature] = 0
                    magpiedata_difference_site2[magpie_feature] = 0
                    magpiedata_composition_average_site3[magpie_feature] = 0
                    magpiedata_arithmetic_average_site3[magpie_feature] = 0
                    magpiedata_max_site3[magpie_feature] = 0
                    magpiedata_min_site3[magpie_feature] = 0
                    magpiedata_difference_site3[magpie_feature] = 0
                    # Couplings between sites
                    magpiedata_composition_average_site1site2[magpie_feature] = 0
                    magpiedata_arithmetic_average_site1site2[magpie_feature] = 0
                    magpiedata_difference_site1site2[magpie_feature] = 0
                    magpiedata_composition_average_site1site3[magpie_feature] = 0
                    magpiedata_arithmetic_average_site1site3[magpie_feature] = 0
                    magpiedata_difference_site1site3[magpie_feature] = 0
                    magpiedata_composition_average_site2site3[magpie_feature] = 0
                    magpiedata_arithmetic_average_site2site3[magpie_feature] = 0
                    magpiedata_difference_site2site3[magpie_feature] = 0

        # Original magpie feature set
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

        # Site-specific magpie features
        if site_dict:
            for element in magpiedata_atomic:
                for site, comp_dict in site_dict.items():
                    magpie_data_by_site_collected = list()
                    for el, amt in comp_dict.items():
                        if el == element:
                            magpie_data_by_site_collected.append(magpiedata_atomic[element])
                    # Here, calc magpie values over the particular site
                    for magpiedata in magpie_data_by_site_collected:
                        for magpie_feature, feature_value in magpiedata.items():
                            if feature_value is not 'NaN':
                                if site == "Site1":
                                    # Composition weighted average by site
                                    magpiedata_composition_average_site1[magpie_feature] += feature_value*float(site_dict[site][element])/site1_total
                                    # Arithmetic average by site
                                    magpiedata_arithmetic_average_site1[magpie_feature] += feature_value / num_site1_elements
                                    # Max features by site
                                    if magpiedata_max_site1[magpie_feature] > 0:
                                        if feature_value > magpiedata_max_site1[magpie_feature]:
                                            magpiedata_max_site1[magpie_feature] = feature_value
                                    elif magpiedata_max_site1[magpie_feature] == 0:
                                        magpiedata_max_site1[magpie_feature] = feature_value
                                    # Min features by site
                                    if magpiedata_min_site1[magpie_feature] > 0:
                                        if feature_value < magpiedata_min_site1[magpie_feature]:
                                            magpiedata_min_site1[magpie_feature] = feature_value
                                    elif magpiedata_min_site1[magpie_feature] == 0:
                                        magpiedata_min_site1[magpie_feature] = feature_value
                                    # Difference features (max - min)
                                    magpiedata_difference_site1[magpie_feature] = magpiedata_max_site1[magpie_feature] - magpiedata_min_site1[magpie_feature]
                                elif site == "Site2":
                                    # Composition weighted average by site
                                    magpiedata_composition_average_site2[magpie_feature] += feature_value*float(site_dict[site][element])/site2_total
                                    # Arithmetic average by site
                                    magpiedata_arithmetic_average_site2[magpie_feature] += feature_value / num_site2_elements
                                    # Max features by site
                                    if magpiedata_max_site2[magpie_feature] > 0:
                                        if feature_value > magpiedata_max_site2[magpie_feature]:
                                            magpiedata_max_site2[magpie_feature] = feature_value
                                    elif magpiedata_max_site2[magpie_feature] == 0:
                                        magpiedata_max_site2[magpie_feature] = feature_value
                                    # Min features by site
                                    if magpiedata_min_site2[magpie_feature] > 0:
                                        if feature_value < magpiedata_min_site2[magpie_feature]:
                                            magpiedata_min_site2[magpie_feature] = feature_value
                                    elif magpiedata_min_site2[magpie_feature] == 0:
                                        magpiedata_min_site2[magpie_feature] = feature_value
                                    # Difference features (max - min)
                                    magpiedata_difference_site2[magpie_feature] = magpiedata_max_site2[magpie_feature] - magpiedata_min_site2[magpie_feature]
                                elif site == "Site3":
                                    # Composition weighted average by site
                                    magpiedata_composition_average_site3[magpie_feature] += feature_value*float(site_dict[site][element])/site3_total
                                    # Arithmetic average by site
                                    magpiedata_arithmetic_average_site3[magpie_feature] += feature_value / num_site3_elements
                                    # Max features by site
                                    if magpiedata_max_site3[magpie_feature] > 0:
                                        if feature_value > magpiedata_max_site3[magpie_feature]:
                                            magpiedata_max_site3[magpie_feature] = feature_value
                                    elif magpiedata_max_site3[magpie_feature] == 0:
                                        magpiedata_max_site3[magpie_feature] = feature_value
                                    # Min features by site
                                    if magpiedata_min_site3[magpie_feature] > 0:
                                        if feature_value < magpiedata_min_site3[magpie_feature]:
                                            magpiedata_min_site3[magpie_feature] = feature_value
                                    elif magpiedata_min_site3[magpie_feature] == 0:
                                        magpiedata_min_site3[magpie_feature] = feature_value
                                    # Difference features (max - min)
                                    magpiedata_difference_site3[magpie_feature] = magpiedata_max_site3[magpie_feature] - magpiedata_min_site3[magpie_feature]
                                    # Add Site couplings here
                                    magpiedata_composition_average_site1site2[magpie_feature] += (magpiedata_composition_average_site1[magpie_feature]+magpiedata_composition_average_site2[magpie_feature])/2
                                    magpiedata_arithmetic_average_site1site2[magpie_feature] += (magpiedata_arithmetic_average_site1[magpie_feature]+magpiedata_arithmetic_average_site2[magpie_feature])/2
                                    #magpiedata_difference_site1site2[magpie_feature] += abs(magpiedata_difference_site1[magpie_feature]-magpiedata_difference_site2[magpie_feature])
                                    magpiedata_difference_site1site2[magpie_feature] += max(magpiedata_max_site1[magpie_feature], magpiedata_max_site2[magpie_feature])-min(magpiedata_min_site1[magpie_feature],magpiedata_min_site2[magpie_feature])
                                    magpiedata_composition_average_site1site3[magpie_feature] += (magpiedata_composition_average_site1[magpie_feature]+magpiedata_composition_average_site3[magpie_feature])/2
                                    magpiedata_arithmetic_average_site1site3[magpie_feature] += (magpiedata_arithmetic_average_site1[magpie_feature]+magpiedata_arithmetic_average_site3[magpie_feature])/2
                                    #magpiedata_difference_site1site3[magpie_feature] += abs(magpiedata_difference_site1[magpie_feature]-magpiedata_difference_site3[magpie_feature])
                                    magpiedata_difference_site1site3[magpie_feature] += max(magpiedata_max_site1[magpie_feature],magpiedata_max_site3[magpie_feature]) - min(magpiedata_min_site1[magpie_feature], magpiedata_min_site3[magpie_feature])
                                    magpiedata_composition_average_site2site3[magpie_feature] += (magpiedata_composition_average_site2[magpie_feature]+magpiedata_composition_average_site3[magpie_feature])/2
                                    magpiedata_arithmetic_average_site2site3[magpie_feature] += (magpiedata_arithmetic_average_site2[magpie_feature]+magpiedata_arithmetic_average_site3[magpie_feature])/2
                                    #magpiedata_difference_site2site3[magpie_feature] += abs(magpiedata_difference_site2[magpie_feature]-magpiedata_difference_site3[magpie_feature])
                                    magpiedata_difference_site2site3[magpie_feature] += max(magpiedata_max_site2[magpie_feature], magpiedata_max_site3[magpie_feature]) - min(magpiedata_min_site2[magpie_feature], magpiedata_min_site3[magpie_feature])
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

        # Rename feature dicts for sublattice specific cases
        magpiedata_composition_average_site1_renamed = {}
        magpiedata_arithmetic_average_site1_renamed = {}
        magpiedata_max_site1_renamed = {}
        magpiedata_min_site1_renamed = {}
        magpiedata_difference_site1_renamed = {}
        magpiedata_composition_average_site2_renamed = {}
        magpiedata_arithmetic_average_site2_renamed = {}
        magpiedata_max_site2_renamed = {}
        magpiedata_min_site2_renamed = {}
        magpiedata_difference_site2_renamed = {}
        magpiedata_composition_average_site3_renamed = {}
        magpiedata_arithmetic_average_site3_renamed = {}
        magpiedata_max_site3_renamed = {}
        magpiedata_min_site3_renamed = {}
        magpiedata_difference_site3_renamed = {}
        # Couplings between sites
        magpiedata_composition_average_site1site2_renamed = {}
        magpiedata_arithmetic_average_site1site2_renamed = {}
        magpiedata_difference_site1site2_renamed = {}
        magpiedata_composition_average_site1site3_renamed = {}
        magpiedata_arithmetic_average_site1site3_renamed = {}
        magpiedata_difference_site1site3_renamed = {}
        magpiedata_composition_average_site2site3_renamed = {}
        magpiedata_arithmetic_average_site2site3_renamed = {}
        magpiedata_difference_site2site3_renamed = {}

        if site_dict:
            if number_sites == 1:
                for key in magpiedata_composition_average_site1:
                    magpiedata_composition_average_site1_renamed["Site1_"+ key + "_composition_average"] = magpiedata_composition_average_site1[key]
                for key in magpiedata_arithmetic_average_site1:
                    magpiedata_arithmetic_average_site1_renamed["Site1_"+ key + "_arithmetic_average"] = magpiedata_arithmetic_average_site1[key]
                for key in magpiedata_max_site1:
                    magpiedata_max_site1_renamed["Site1_"+ key + "_max_value"] = magpiedata_max_site1[key]
                for key in magpiedata_min_site1:
                    magpiedata_min_site1_renamed["Site1_"+ key + "_min_value"] = magpiedata_min_site1[key]
                for key in magpiedata_difference_site1:
                    magpiedata_difference_site1_renamed["Site1_"+ key + "_difference"] = magpiedata_difference_site1[key]
            elif number_sites == 2:
                for key in magpiedata_composition_average_site1:
                    magpiedata_composition_average_site1_renamed["Site1_"+ key + "_composition_average"] = magpiedata_composition_average_site1[key]
                for key in magpiedata_arithmetic_average_site1:
                    magpiedata_arithmetic_average_site1_renamed["Site1_"+ key + "_arithmetic_average"] = magpiedata_arithmetic_average_site1[key]
                for key in magpiedata_max_site1:
                    magpiedata_max_site1_renamed["Site1_"+ key + "_max_value"] = magpiedata_max_site1[key]
                for key in magpiedata_min_site1:
                    magpiedata_min_site1_renamed["Site1_"+ key + "_min_value"] = magpiedata_min_site1[key]
                for key in magpiedata_difference_site1:
                    magpiedata_difference_site1_renamed["Site1_"+ key + "_difference"] = magpiedata_difference_site1[key]
                for key in magpiedata_composition_average_site2:
                    magpiedata_composition_average_site2_renamed["Site2_"+ key + "_composition_average"] = magpiedata_composition_average_site2[key]
                for key in magpiedata_arithmetic_average_site2:
                    magpiedata_arithmetic_average_site2_renamed["Site2_"+ key + "_arithmetic_average"] = magpiedata_arithmetic_average_site2[key]
                for key in magpiedata_max_site2:
                    magpiedata_max_site2_renamed["Site2_"+ key + "_max_value"] = magpiedata_max_site2[key]
                for key in magpiedata_min_site2:
                    magpiedata_min_site2_renamed["Site2_"+ key + "_min_value"] = magpiedata_min_site2[key]
                for key in magpiedata_difference_site2:
                    magpiedata_difference_site2_renamed["Site2_"+ key + "_difference"] = magpiedata_difference_site2[key]
            elif number_sites == 3:
                for key in magpiedata_composition_average_site1:
                    magpiedata_composition_average_site1_renamed["Site1_"+ key + "_composition_average"] = magpiedata_composition_average_site1[key]
                for key in magpiedata_arithmetic_average_site1:
                    magpiedata_arithmetic_average_site1_renamed["Site1_"+ key + "_arithmetic_average"] = magpiedata_arithmetic_average_site1[key]
                for key in magpiedata_max_site1:
                    magpiedata_max_site1_renamed["Site1_"+ key + "_max_value"] = magpiedata_max_site1[key]
                for key in magpiedata_min_site1:
                    magpiedata_min_site1_renamed["Site1_"+ key + "_min_value"] = magpiedata_min_site1[key]
                for key in magpiedata_difference_site1:
                    magpiedata_difference_site1_renamed["Site1_"+ key + "_difference"] = magpiedata_difference_site1[key]
                for key in magpiedata_composition_average_site2:
                    magpiedata_composition_average_site2_renamed["Site2_"+ key + "_composition_average"] = magpiedata_composition_average_site2[key]
                for key in magpiedata_arithmetic_average_site2:
                    magpiedata_arithmetic_average_site2_renamed["Site2_"+ key + "_arithmetic_average"] = magpiedata_arithmetic_average_site2[key]
                for key in magpiedata_max_site2:
                    magpiedata_max_site2_renamed["Site2_"+ key + "_max_value"] = magpiedata_max_site2[key]
                for key in magpiedata_min_site2:
                    magpiedata_min_site2_renamed["Site2_"+ key + "_min_value"] = magpiedata_min_site2[key]
                for key in magpiedata_difference_site2:
                    magpiedata_difference_site2_renamed["Site2_"+ key + "_difference"] = magpiedata_difference_site2[key]
                for key in magpiedata_composition_average_site3:
                    magpiedata_composition_average_site3_renamed["Site3_"+ key + "_composition_average"] = magpiedata_composition_average_site3[key]
                for key in magpiedata_arithmetic_average_site3:
                    magpiedata_arithmetic_average_site3_renamed["Site3_"+ key + "_arithmetic_average"] = magpiedata_arithmetic_average_site3[key]
                for key in magpiedata_max_site3:
                    magpiedata_max_site3_renamed["Site3_"+ key + "_max_value"] = magpiedata_max_site3[key]
                for key in magpiedata_min_site1:
                    magpiedata_min_site3_renamed["Site3_"+ key + "_min_value"] = magpiedata_min_site3[key]
                for key in magpiedata_difference_site3:
                    magpiedata_difference_site3_renamed["Site3_"+ key + "_difference"] = magpiedata_difference_site3[key]
                # Couplings between sites
                for key in magpiedata_composition_average_site1site2:
                    magpiedata_composition_average_site1site2_renamed["Site1Site2_"+ key + "_composition_average"] = magpiedata_composition_average_site1site2[key]
                for key in magpiedata_arithmetic_average_site1site2:
                    magpiedata_arithmetic_average_site1site2_renamed["Site1Site2_" + key + "_arithmetic_average"] = magpiedata_arithmetic_average_site1site2[key]
                for key in magpiedata_difference_site1site2:
                    magpiedata_difference_site1site2_renamed["Site1Site2_" + key + "_difference"] = magpiedata_difference_site1site2[key]
                for key in magpiedata_composition_average_site1site3:
                    magpiedata_composition_average_site1site3_renamed["Site1Site3_"+ key + "_composition_average"] = magpiedata_composition_average_site1site3[key]
                for key in magpiedata_arithmetic_average_site1site3:
                    magpiedata_arithmetic_average_site1site3_renamed["Site1Site3_" + key + "_arithmetic_average"] = magpiedata_arithmetic_average_site1site3[key]
                for key in magpiedata_difference_site1site3:
                    magpiedata_difference_site1site3_renamed["Site1Site3_" + key + "_difference"] = magpiedata_difference_site1site3[key]
                for key in magpiedata_composition_average_site2site3:
                    magpiedata_composition_average_site2site3_renamed["Site2Site3_"+ key + "_composition_average"] = magpiedata_composition_average_site2site3[key]
                for key in magpiedata_arithmetic_average_site2site3:
                    magpiedata_arithmetic_average_site2site3_renamed["Site2Site3_" + key + "_arithmetic_average"] = magpiedata_arithmetic_average_site2site3[key]
                for key in magpiedata_difference_site2site3:
                    magpiedata_difference_site2site3_renamed["Site2Site3_" + key + "_difference"] = magpiedata_difference_site2site3[key]
        if site_dict:
            if number_sites == 1:
                return (magpiedata_composition_average_renamed, magpiedata_arithmetic_average_renamed,
                        magpiedata_max_renamed, magpiedata_min_renamed, magpiedata_difference_renamed,
                        magpiedata_composition_average_site1_renamed, magpiedata_arithmetic_average_site1_renamed,
                        magpiedata_max_site1_renamed, magpiedata_min_site1_renamed, magpiedata_difference_site1_renamed)
            elif number_sites == 2:
                return (magpiedata_composition_average_renamed, magpiedata_arithmetic_average_renamed,
                        magpiedata_max_renamed, magpiedata_min_renamed, magpiedata_difference_renamed,
                        magpiedata_composition_average_site1_renamed, magpiedata_arithmetic_average_site1_renamed,
                        magpiedata_max_site1_renamed, magpiedata_min_site1_renamed, magpiedata_difference_site1_renamed,
                        magpiedata_composition_average_site2_renamed, magpiedata_arithmetic_average_site2_renamed,
                        magpiedata_max_site2_renamed, magpiedata_min_site2_renamed, magpiedata_difference_site2_renamed)
            elif number_sites == 3:
                return (magpiedata_composition_average_renamed, magpiedata_arithmetic_average_renamed,
                        magpiedata_max_renamed, magpiedata_min_renamed, magpiedata_difference_renamed,
                        magpiedata_composition_average_site1_renamed, magpiedata_arithmetic_average_site1_renamed,
                        magpiedata_max_site1_renamed, magpiedata_min_site1_renamed, magpiedata_difference_site1_renamed,
                        magpiedata_composition_average_site2_renamed, magpiedata_arithmetic_average_site2_renamed,
                        magpiedata_max_site2_renamed, magpiedata_min_site2_renamed, magpiedata_difference_site2_renamed,
                        magpiedata_composition_average_site3_renamed, magpiedata_arithmetic_average_site3_renamed,
                        magpiedata_max_site3_renamed, magpiedata_min_site3_renamed, magpiedata_difference_site3_renamed,
                        magpiedata_composition_average_site1site2_renamed, magpiedata_arithmetic_average_site1site2_renamed, magpiedata_difference_site1site2_renamed,
                        magpiedata_composition_average_site1site3_renamed, magpiedata_arithmetic_average_site1site3_renamed, magpiedata_difference_site1site3_renamed,
                        magpiedata_composition_average_site2site3_renamed, magpiedata_arithmetic_average_site2site3_renamed, magpiedata_difference_site2site3_renamed)
        else:
            return (magpiedata_composition_average_renamed, magpiedata_arithmetic_average_renamed,
                    magpiedata_max_renamed, magpiedata_min_renamed, magpiedata_difference_renamed)

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

# Old Citrine classes likely to be deleted
"""
    class Citrine(BaseEstimator, TransformerMixin):


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
    """

"""
class CitrineFeatureGeneration(object):

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
"""
