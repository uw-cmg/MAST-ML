__author__ = 'Ryan Jacobs'

import pandas as pd
import sys
import os
import logging
from pymatgen import Element, Composition
from pymatgen.matproj.rest import MPRester
from citrination_client import *
from DataOperations import DataframeUtilities

class MagpieFeatureGeneration(object):
    """Class to generate new features using Magpie data and dataframe containing material compositions. Creates
     a dataframe and append features to existing feature dataframes
    """
    def __init__(self, configdict, dataframe):
        self.configdict = configdict
        self.dataframe = dataframe

    @property
    def get_original_dataframe(self):
        return self.dataframe

    def generate_magpie_features(self, save_to_csv=True):
        compositions = []
        composition_components = []

        # Replace empty composition fields with empty string instead of NaN
        self.dataframe = self.dataframe.fillna('')
        for column in self.dataframe.columns:
            if 'Material composition' in column:
                composition_components.append(self.dataframe[column].tolist())

        if len(composition_components) < 1:
            logging.info('ERROR: No column with "Material composition xx" was found in the supplied dataframe')
            sys.exit()

        row = 0
        while row < len(composition_components[0]):
            composition = ''
            for composition_component in composition_components:
                composition += str(composition_component[row])
            compositions.append(composition)
            row += 1

        # Add the column of combined material compositions into the dataframe
        self.dataframe['Material compositions'] = compositions

        # Assign each magpiedata feature set to appropriate composition name
        magpiedata_dict_composition_average = {}
        magpiedata_dict_arithmetic_average = {}
        magpiedata_dict_max = {}
        magpiedata_dict_min = {}
        magpiedata_dict_difference = {}
        magpiedata_dict_atomic_bysite = {}

        for composition in compositions:
            magpiedata_composition_average, magpiedata_arithmetic_average, magpiedata_max, magpiedata_min, magpiedata_difference = self._get_computed_magpie_features(composition=composition)
            magpiedata_atomic_notparsed = self._get_atomic_magpie_features(composition=composition)

            magpiedata_dict_composition_average[composition] = magpiedata_composition_average
            magpiedata_dict_arithmetic_average[composition] = magpiedata_arithmetic_average
            magpiedata_dict_max[composition] = magpiedata_max
            magpiedata_dict_min[composition] = magpiedata_min
            magpiedata_dict_difference[composition] = magpiedata_difference

            # Add site-specific elemental features
            count = 1
            magpiedata_atomic_bysite = {}
            for entry in magpiedata_atomic_notparsed.keys():
                for magpiefeature, featurevalue in magpiedata_atomic_notparsed[entry].items():
                    magpiedata_atomic_bysite["Site"+str(count)+"_"+str(magpiefeature)] = featurevalue
                count += 1

            magpiedata_dict_atomic_bysite[composition] = magpiedata_atomic_bysite

        magpiedata_dict_list = [magpiedata_dict_composition_average, magpiedata_dict_arithmetic_average,
                                magpiedata_dict_max, magpiedata_dict_min, magpiedata_dict_difference, magpiedata_dict_atomic_bysite]

        for magpiedata_dict in magpiedata_dict_list:
            dataframe_magpie = pd.DataFrame.from_dict(data=magpiedata_dict, orient='index')
            # Need to reorder compositions in new dataframe to match input dataframe
            dataframe_magpie = dataframe_magpie.reindex(self.dataframe['Material compositions'].tolist())
            # Need to make compositions the first column, instead of the row names
            dataframe_magpie.index.name = 'Material compositions'
            dataframe_magpie.reset_index(inplace=True)
            # Need to delete duplicate column before merging dataframes
            del dataframe_magpie['Material compositions']
            # Merge magpie feature dataframe with originally supplied dataframe
            dataframe = DataframeUtilities()._merge_dataframe_columns(dataframe1=self.dataframe, dataframe2=dataframe_magpie)

        if save_to_csv == bool(True):
            # Get y_feature in this dataframe, attach it to save path
            for column in dataframe.columns.values:
                if column in self.configdict['General Setup']['target_feature']:
                    filetag = column
            dataframe.to_csv(self.configdict['General Setup']['save_path']+"/"+'input_with_magpie_features'+'_'+str(filetag)+'.csv', index=False)

        return dataframe

    def _get_computed_magpie_features(self, composition):
        magpiedata_composition_average = {}
        magpiedata_arithmetic_average = {}
        magpiedata_max = {}
        magpiedata_min = {}
        magpiedata_difference = {}
        magpiedata_atomic = self._get_atomic_magpie_features(composition=composition)
        composition = Composition(composition)
        element_list, atoms_per_formula_unit = self._get_element_list(composition=composition)

        # Initialize feature values to all be 0, because need to dynamically update them with weighted values in next loop.
        for magpie_feature in magpiedata_atomic[element_list[0]].keys():
            magpiedata_composition_average[magpie_feature] = 0
            magpiedata_arithmetic_average[magpie_feature] = 0
            magpiedata_max[magpie_feature] = 0
            magpiedata_min[magpie_feature] = 0
            magpiedata_difference[magpie_feature] = 0

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
                    # Difference features (max - min)
                    magpiedata_difference[magpie_feature] = magpiedata_max[magpie_feature] - magpiedata_min[magpie_feature]

        # Change names of features to reflect each computed type of magpie feature (max, min, etc.)
        magpiedata_composition_average_renamed = {}
        magpiedata_arithmetic_average_renamed = {}
        magpiedata_max_renamed = {}
        magpiedata_min_renamed = {}
        magpiedata_difference_renamed = {}
        for key in magpiedata_composition_average.keys():
            magpiedata_composition_average_renamed[key+"_composition_average"] = magpiedata_composition_average[key]
        for key in magpiedata_arithmetic_average.keys():
            magpiedata_arithmetic_average_renamed[key+"_arithmetic_average"] = magpiedata_arithmetic_average[key]
        for key in magpiedata_max.keys():
            magpiedata_max_renamed[key+"_max_value"] = magpiedata_max[key]
        for key in magpiedata_min.keys():
            magpiedata_min_renamed[key+"_min_value"] = magpiedata_min[key]
        for key in magpiedata_difference.keys():
            magpiedata_difference_renamed[key+"_difference"] = magpiedata_difference[key]

        return magpiedata_composition_average_renamed, magpiedata_arithmetic_average_renamed, magpiedata_max_renamed, magpiedata_min_renamed, magpiedata_difference_renamed

    def _get_atomic_magpie_features(self, composition):
        # Get .table files containing feature values for each element, assign file names as feature names
        config_files_path = self.configdict['General Setup']['config_files_path']
        data_path = config_files_path+'/magpiedata/magpie_elementdata'
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

class MaterialsProjectFeatureGeneration(object):
    """Class to generate new features using the Materials Project and dataframe containing material compositions. Creates
     a dataframe and append features to existing feature dataframes
    """
    def __init__(self, dataframe, mapi_key):
        self.dataframe = dataframe
        self.mapi_key = mapi_key

    def generate_materialsproject_features(self, save_to_csv=True):
        try:
            compositions = self.dataframe['Material compositions']
        except KeyError:
            print('No column called "Material compositions" exists in the supplied dataframe.')
            sys.exit()

        mpdata_dict_composition = {}
        for composition in compositions:
            composition_data_mp = self._get_data_from_materials_project(composition=composition)
            mpdata_dict_composition[composition] = composition_data_mp

        dataframe = self.dataframe
        dataframe_mp = pd.DataFrame.from_dict(data=mpdata_dict_composition, orient='index')
        # Need to reorder compositions in new dataframe to match input dataframe
        dataframe_mp = dataframe_mp.reindex(self.dataframe['Material compositions'].tolist())
        # Need to make compositions the first column, instead of the row names
        dataframe_mp.index.name = 'Material compositions'
        dataframe_mp.reset_index(inplace=True)
        # Need to delete duplicate column before merging dataframes
        del dataframe_mp['Material compositions']
        # Merge magpie feature dataframe with originally supplied dataframe
        dataframe = DataframeUtilities()._merge_dataframe_columns(dataframe1=dataframe, dataframe2=dataframe_mp)
        if save_to_csv == bool(True):
            dataframe.to_csv('input_with_materialsproject_features.csv', index=False)
        return dataframe

    def _get_data_from_materials_project(self, composition):
        mprester = MPRester(self.mapi_key)
        structure_data_list = mprester.get_data(chemsys_formula_id=composition)

        # Sort structures by stability (i.e. E above hull), and only return most stable compound data
        if len(structure_data_list) > 0:
            structure_data_list = sorted(structure_data_list, key= lambda e_above: e_above['e_above_hull'])
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
            for property in property_list:
                if property in elastic_property_list:
                    try:
                        structure_data_dict_condensed[property] = structure_data_most_stable["elasticity"][property]
                    except TypeError:
                        structure_data_dict_condensed[property] = ''
                elif property == "number":
                    try:
                        structure_data_dict_condensed["Spacegroup_"+property] = structure_data_most_stable["spacegroup"][property]
                    except TypeError:
                        structure_data_dict_condensed[property] = ''
                else:
                    try:
                        structure_data_dict_condensed[property] = structure_data_most_stable[property]
                    except TypeError:
                        structure_data_dict_condensed[property] = ''
        else:
            for property in property_list:
                if property == "number":
                    structure_data_dict_condensed["Spacegroup_"+property] = ''
                else:
                    structure_data_dict_condensed[property] = ''

        return structure_data_dict_condensed

class CitrineFeatureGeneration(object):
    """THIS CLASS IS UNDER CONSTRUCTION AS OF 5/31/17. WILL FINISH ASAP

    Class to generate new features using the Citrination client and dataframe containing material compositions. Creates
     a dataframe and append features to existing feature dataframes
    """
    def __init__(self, dataframe, api_key):
        self.dataframe = dataframe
        self.api_key = api_key
        self.client = CitrinationClient(api_key, 'https://citrination.com')

    def generate_citrine_features(self, save_to_csv=True):
        pass

    def _get_data_from_citrine(self, composition):
        #pif_query = PifQuery(system=SystemQuery(chemical_formula=ChemicalFieldOperation(filter=ChemicalFilter(equal=formula)),
        #                            properties=PropertyQuery(name=FieldOperation(filter=Filter(equal=property)),
        #                            value=FieldOperation(filter=Filter(min=min_measurement, max=max_measurement)),
        #                            data_type=FieldOperation(filter=Filter(equal=data_type))),
        #                            references=ReferenceQuery(doi=FieldOperation(filter=Filter(equal=reference)))),
        #                            include_datasets=[data_set_id], from_index=start, size=per_page)

        pif_query = PifQuery(system=SystemQuery(chemical_formula=ChemicalFieldOperation(filter=ChemicalFilter(equal=composition))))

        # Check if any results found
        if 'hits' not in self.client.search(pif_query).as_dictionary():
            raise KeyError('No results found!')
        data = self.client.search(pif_query).as_dictionary()['hits']
        return data