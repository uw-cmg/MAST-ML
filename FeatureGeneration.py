__author__ = 'Ryan Jacobs'

import pandas as pd
import sys
import os
import logging
from pymatgen import Element, Composition
from pymatgen.ext.matproj import MPRester
from citrination_client import PifQuery, SystemQuery, ChemicalFieldQuery, ChemicalFilter
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

        dataframe = self.dataframe
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
            dataframe = DataframeUtilities()._merge_dataframe_columns(dataframe1=dataframe, dataframe2=dataframe_magpie)

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
    def __init__(self, configdict, dataframe, mapi_key):
        self.configdict = configdict
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
            # Get y_feature in this dataframe, attach it to save path
            for column in dataframe.columns.values:
                if column in self.configdict['General Setup']['target_feature']:
                    filetag = column
            dataframe.to_csv(self.configdict['General Setup']['save_path']+"/"+'input_with_matproj_features'+'_'+str(filetag)+'.csv', index=False)
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

    def __init__(self, configdict, dataframe, api_key):
        self.configdict = configdict
        self.dataframe = dataframe
        self.api_key = api_key
        self.client = CitrinationClient(api_key, 'https://citrination.com')

    def generate_citrine_features(self, save_to_csv=True):
        logging.info('WARNING: You have specified generation of features from Citrine. Based on which materials you are'
                     'interested in, there may be many records to parse through, thus this routine may take a long time to complete!')
        compositions = self.dataframe['Material compositions'].tolist()
        citrine_dict_property_min = dict()
        citrine_dict_property_max = dict()
        citrine_dict_property_avg = dict()
        for composition in compositions:
            pifquery = self._get_pifquery(composition=composition)
            property_name_list, property_value_list = self._get_pifquery_property_list(pifquery=pifquery)
            property_names_unique, parsed_property_min, parsed_property_max, parsed_property_avg = self._parse_pifquery_property_list(property_name_list=property_name_list, property_value_list=property_value_list)
            citrine_dict_property_min[composition] = parsed_property_min
            citrine_dict_property_max[composition] = parsed_property_max
            citrine_dict_property_avg[composition] = parsed_property_avg

        dataframe = self.dataframe
        citrine_dict_list = [citrine_dict_property_min, citrine_dict_property_max, citrine_dict_property_avg]
        for citrine_dict in citrine_dict_list:
            dataframe_citrine = pd.DataFrame.from_dict(data=citrine_dict, orient='index')
            # Need to reorder compositions in new dataframe to match input dataframe
            dataframe_citrine = dataframe_citrine.reindex(self.dataframe['Material compositions'].tolist())
            # Need to make compositions the first column, instead of the row names
            dataframe_citrine.index.name = 'Material compositions'
            dataframe_citrine.reset_index(inplace=True)
            # Need to delete duplicate column before merging dataframes
            del dataframe_citrine['Material compositions']
            # Merge magpie feature dataframe with originally supplied dataframe
            dataframe = DataframeUtilities()._merge_dataframe_columns(dataframe1=dataframe, dataframe2=dataframe_citrine)

        if save_to_csv == bool(True):
            # Get y_feature in this dataframe, attach it to save path
            for column in dataframe.columns.values:
                if column in self.configdict['General Setup']['target_feature']:
                    filetag = column
            dataframe.to_csv(self.configdict['General Setup']['save_path']+"/"+'input_with_citrine_features'+'_'+str(filetag)+'.csv', index=False)

        return dataframe

    def _get_pifquery(self, composition):
        pif_query = PifQuery(system=SystemQuery(chemical_formula=ChemicalFieldQuery(filter=ChemicalFilter(equal=composition))))
        # Check if any results found
        if 'hits' not in self.client.search(pif_query).as_dictionary():
            raise KeyError('No results found!')
        pifquery = self.client.search(pif_query).as_dictionary()['hits']
        return pifquery

    def _get_pifquery_property_list(self, pifquery):
        property_name_list = list()
        property_value_list = list()
        accepted_properties_list = ['mass', 'space group', 'band', 'Band', 'energy', 'volume', 'density', 'dielectric',
                                    'Dielectric',
                                    'Enthalpy', 'Convex', 'Magnetization', 'Elements', 'Modulus', 'Shear', "Poisson's",
                                    'Elastic', 'Energy']
        for result_number, results in enumerate(pifquery):
            for system_heading, system_value in results.items():
                if system_heading == 'system':
                    # print('FOUND SYSTEM')
                    for property_name, property_value in system_value.items():
                        if property_name == 'properties':
                            # print('FOUND PROPERTIES')
                            # pprint(property_value)
                            for list_index, list_element in enumerate(property_value):
                                for name, value in property_value[list_index].items():
                                    if name == 'name':
                                        # Check that the property name is in the acceptable property list
                                        if value != "CIF File":
                                            for entry in accepted_properties_list:
                                                if entry in value:
                                                    # print('found acceptable name', entry, 'for name', value, 'with value',property_value[list_index]['scalars'][0]['value'] )
                                                    property_name_list.append(value)
                                                    try:
                                                        property_value_list.append(
                                                            float(property_value[list_index]['scalars'][0]['value']))
                                                    except (ValueError, KeyError):
                                                        # print('found something to remove', property_value[list_index]['scalars'][0]['value'])
                                                        property_name_list.pop(-1)
                                                        continue
        return property_name_list, property_value_list

    def _parse_pifquery_property_list(self, property_name_list, property_value_list):
        parsed_property_max = dict()
        parsed_property_min = dict()
        parsed_property_avg = dict()
        property_names_unique = list()
        if len(property_name_list) != len(property_value_list):
            print('Error! Length of property name and property value lists are not the same. There must be a bug in the _get_pifquerey_property_list method')
            sys.exit()
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