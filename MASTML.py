__author__ = 'Ryan Jacobs, Tam Mayeshiba'

import sys
import os
from MASTMLInitializer import MASTMLWrapper, ConfigFileValidator
from DataOperations import DataParser, DataframeUtilities
from FeatureGeneration import MagpieFeatureGeneration, MaterialsProjectFeatureGeneration, CitrineFeatureGeneration
from FeatureOperations import FeatureNormalization, FeatureIO, MiscFeatureOperations
from FeatureSelection import FeatureSelection, DimensionalReduction, MiscFeatureSelectionOperations
import logging
import shutil
import time
from custom_features import cf_help
import matplotlib
env_display = os.getenv("DISPLAY")
if env_display is None:
    matplotlib.use("agg")
from DataHandler import DataHandler
import importlib
import pandas as pd
from SingleFit import timeit

class MASTMLDriver(object):

    def __init__(self, configfile):
        self.configfile = configfile
        #Set in code
        self.general_setup = None
        self.csv_setup = None
        self.data_setup = None
        self.models_and_tests_setup = None
        self.configdict = None
        self.mastmlwrapper = None
        self.save_path = None #top-level save path
        self.data_dict = dict() #Data dictionary
        self.model_list = list() #list of actual models
        self.model_vals = list() #list of model names, like "gkrr_model"
        self.param_optimizing_tests = ["ParamOptGA"]
        self.readme_html = list()
        self.readme_html_tests = list()
        self.start_time = None
        self.favorites_dict=dict()
        self.test_save_paths=list()
        return

    # This will later be removed as the parsed input file should have values all containing correct datatype
    def string_or_list_input_to_list(self, unknown_input_val):
        input_list=list()
        if type(unknown_input_val) is str:
            input_list.append(unknown_input_val)
        elif type(unknown_input_val) is list:
            for unknown_input in unknown_input_val:
                input_list.append(unknown_input)
        return input_list

    def run_MASTML(self):
        # Begin MASTML session
        self._initialize_mastml_session()
        self._initialize_html()
        self.set_favorites_dict()

        # Parse MASTML input file
        self.mastmlwrapper, self.configdict, errors_present = self._generate_mastml_wrapper()
        self.data_setup = self.mastmlwrapper.process_config_keyword(keyword='Data Setup')

        # General setup
        self._perform_general_setup()

        # Perform CSV setup (optional)
        if "CSV Setup" in self.configdict.keys():
            self._perform_csv_setup()

        self.data_dict = self._create_data_dict()

        # Gather models
        (self.model_list, self.model_vals) = self._gather_models()

        # Gather tests
        test_list = self._gather_tests(mastmlwrapper=self.mastmlwrapper, configdict=self.configdict, data_dict=self.data_dict, model_list=self.model_list, save_path=self.save_path,
                model_vals = self.model_vals)

        # End MASTML session
        self._move_log_and_input_files()
        self._end_html()

        return

    def _initialize_mastml_session(self):
        logging.basicConfig(filename='MASTMLlog.log', level='INFO')
        current_time = time.strftime('%Y'+'-'+'%m'+'-'+'%d'+', '+'%H'+' hours, '+'%M'+' minutes, '+'and '+'%S'+' seconds')
        logging.info('Initiated new MASTML session at: %s' % current_time)
        self.start_time = time.strftime("%Y-%m-%d, %H:%M:%S")
        return

    def _initialize_html(self):
        self.readme_html.append("<HTML>\n")
        self.readme_html.append("<TITLE>MASTML</TITLE>\n")
        self.readme_html.append("<BODY>\n")
        self.readme_html.append("<H1>%s</H1>\n" % "MAST Machine Learning Output")
        self.readme_html.append("%s<BR>\n" % self.start_time)
        self.readme_html.append("<HR>\n")
        return

    def _end_html(self):
        self.readme_html.append("<HR>\n")
        self.readme_html.append("<H2>Setup</H2>\n")
        logpath = os.path.join(self.save_path, "MASTMLlog.log")
        logpath = os.path.relpath(logpath, self.save_path)
        self.readme_html.append('<A HREF="%s">Log file</A><BR>\n' % logpath)
        confpath = os.path.join(self.save_path, str(self.configfile))
        confpath = os.path.relpath(confpath, self.save_path)
        self.readme_html.append('<A HREF="%s">Config file</A><BR>\n' % confpath)
        self.readme_html.append("<HR>\n")
        self.readme_html.append("</BODY>\n")
        self.readme_html.append("</HTML>\n")
        with open(os.path.join(self.save_path, "index.html"),"w") as hfile:
            hfile.writelines(self.readme_html)
            hfile.writelines(self.readme_html_tests)
        return

    def _generate_mastml_wrapper(self):
        configdict, errors_present = ConfigFileValidator(configfile=self.configfile).run_config_validation()
        mastmlwrapper = MASTMLWrapper(configdict=configdict)
        logging.info('Successfully read in and parsed your MASTML input file, %s' % str(self.configfile))
        return mastmlwrapper, configdict, errors_present

    def _perform_general_setup(self):
        self.general_setup = self.mastmlwrapper.process_config_keyword(keyword='General Setup')
        self.save_path = os.path.abspath(self.general_setup['save_path'])
        if not os.path.isdir(self.save_path):
            os.mkdir(self.save_path)
        return self.save_path

    def _perform_csv_setup(self):
        self.csv_setup = self.mastmlwrapper.process_config_keyword(keyword = "CSV Setup")
        setup_class = self.csv_setup.pop("setup_class") #also remove from dict
        class_name = setup_class.split(".")[-1]
        test_module = importlib.import_module('%s' % (setup_class))
        test_class_def = getattr(test_module, class_name)
        logging.debug("Parameters passed by keyword:")
        logging.debug(self.csv_setup)
        test_class = test_class_def(**self.csv_setup)
        test_class.run() 
        return

    def _parse_input_data(self, data_path=""):
        if not(os.path.isfile(data_path)):
            raise OSError("No file found at %s" % data_path)

        Xdata, ydata, x_features, y_feature, dataframe = DataParser(configdict=self.configdict).parse_fromfile(datapath=data_path, as_array=False)
        # Remove dataframe entries that contain 'NaN'

        #print('during import')
        #print(len(x_features))
        #print(dataframe.shape)

        #dataframe = dataframe.dropna()

        #print('during import, after drop')
        #print(len(x_features))
        #print(dataframe.shape)

        return Xdata, ydata, x_features, y_feature, dataframe

    def _create_data_dict(self):
        data_dict=dict()
        for data_name in self.data_setup.keys():
            data_path = self.data_setup[data_name]['data_path']
            data_weights = self.data_setup[data_name]['weights']
            if 'labeling_features' in self.general_setup.keys():
                labeling_features = self.string_or_list_input_to_list(self.general_setup['labeling_features'])
            else:
                labeling_features = None
            if 'target_error_feature' in self.general_setup.keys():
                target_error_feature = self.general_setup['target_error_feature']
            else:
                target_error_feature = None
            if 'grouping_feature' in self.general_setup.keys():
                grouping_feature = self.general_setup['grouping_feature']
            else:
                grouping_feature = None

            if 'Feature Generation' in self.configdict.keys():
                if self.configdict['Feature Generation']['perform_feature_generation'] == bool(True) or self.configdict['Feature Generation']['perform_feature_generation'] == "True":
                    generate_features = True
                else:
                    generate_features = False
            else:
                generate_features = False

            if 'Feature Normalization' in self.configdict.keys():
                if self.configdict['Feature Normalization']['normalize_x_features'] == bool(True) or self.configdict['Feature Normalization']['normalize_x_features'] == "True":
                    normalize_x_features = True
                else:
                    normalize_x_features = False
                if self.configdict['Feature Normalization']['normalize_y_feature'] == bool(True) or self.configdict['Feature Normalization']['normalize_y_feature'] == "True":
                    normalize_y_feature = True
                else:
                    normalize_y_feature = False
            else:
                normalize_x_features = False
                normalize_y_feature = False

            if 'Feature Selection' in self.configdict.keys():
                if self.configdict['Feature Selection']['perform_feature_selection'] == bool(True) or self.configdict['Feature Selection']['perform_feature_selection'] == "True":
                    select_features = True
                else:
                    select_features = False
            else:
                select_features = False

            logging.info("Feature Generation: %s" % generate_features)
            logging.info("Feature Normalization (x_features): %s" % normalize_x_features)
            logging.info("Feature Normalization (y_feature): %s" % normalize_y_feature)
            logging.info("Feature Selection: %s" % select_features)
            # Parse input data file
            Xdata, ydata, x_features, y_feature, dataframe = self._parse_input_data(data_path)

            #print('after import')
            #print(len(x_features))
            #print(dataframe.shape)

            original_x_features = list(x_features)
            original_columns = list(dataframe.columns)
            # Remove any missing rows from dataframe
            #dataframe = dataframe.dropna()

            #print('after import and drop')
            #print(len(x_features))
            #print(dataframe.shape)
            
            # Save off label and grouping data
            dataframe_labeled = pd.DataFrame()
            dataframe_grouped = pd.DataFrame()
            if not (labeling_features is None):
                dataframe_labeled = FeatureIO(dataframe=dataframe).keep_custom_features(features_to_keep=labeling_features, y_feature=y_feature)
                if normalize_x_features == bool(True):
                    dataframe_labeled, scaler = FeatureNormalization(dataframe=dataframe_labeled).normalize_features(x_features=labeling_features, y_feature=y_feature)
            if not (grouping_feature is None):
                dataframe_grouped = FeatureIO(dataframe=dataframe).keep_custom_features(features_to_keep=[grouping_feature], y_feature=y_feature)

            # Generate additional descriptors, as specified in input file (optional)
            if generate_features:
                dataframe = self._perform_feature_generation(dataframe=dataframe)
                # Actually, the x_features_NOUSE is required if starting from no features and doing feature generation. Not renaming for now. RJ 7/17
                Xdata, ydata, x_features_NOUSE, y_feature, dataframe = DataParser(configdict=self.configdict).parse_fromdataframe(dataframe=dataframe, target_feature=y_feature)
                #print('after gen')
                #print(dataframe.shape)
                #print(len(x_features_NOUSE))

            else:
                Xdata, ydata, x_features, y_feature, dataframe = DataParser(configdict=self.configdict).parse_fromdataframe(dataframe=dataframe, target_feature=y_feature)




            # First remove features containing strings before doing feature normalization or other operations, but don't remove grouping features
            if generate_features == bool(True):
                x_features, dataframe_nostrings = MiscFeatureOperations(configdict=self.configdict).remove_features_containing_strings(dataframe=dataframe, x_features=x_features_NOUSE)
                #Remove columns containing all entries of NaN
                dataframe_nostrings = dataframe_nostrings.dropna(axis=1, how='all')
                # Re-obtain x_feature list as some features may have been dropped
                Xdata, ydata, x_features_NOUSE, y_feature, dataframe_nostrings = DataParser(configdict=self.configdict).parse_fromdataframe(dataframe=dataframe_nostrings,target_feature=y_feature)
            else:
                x_features, dataframe_nostrings = MiscFeatureOperations(configdict=self.configdict).remove_features_containing_strings(dataframe=dataframe, x_features=x_features)

            #print('after string remove')
            #print(len(x_features))
            #print(dataframe_nostrings.shape)

            # Remove columns containing all entries of NaN
            dataframe_nostrings = dataframe_nostrings.dropna(axis=1, how='all')

            #print('after dropna')
            #print(len(x_features))
            #print(dataframe_nostrings.shape)

            # Re-obtain x_feature list as some features may have been dropped
            Xdata, ydata, x_features, y_feature, dataframe_nostrings = DataParser(configdict=self.configdict).parse_fromdataframe(dataframe=dataframe_nostrings, target_feature=y_feature)



            logging.debug("pre-changes:%s" % dataframe_nostrings.columns)

            # Normalize features (optional)
            if normalize_x_features == bool(True) or normalize_y_feature == bool(True):
                fn = FeatureNormalization(dataframe=dataframe_nostrings)
                dataframe_nostrings, scaler = fn.normalize_features(x_features=x_features, y_feature=y_feature, normalize_x_features=normalize_x_features, normalize_y_feature=normalize_y_feature)
                x_features, y_feature = DataParser(configdict=self.configdict).get_features(dataframe=dataframe_nostrings, target_feature=y_feature)

            # Perform feature selection and dimensional reduction, as specified in the input file (optional)
            if (select_features == bool(True)) and (y_feature in dataframe_nostrings.columns):
                # Remove any additional columns that are not x_features using to be fit to data
                features = dataframe_nostrings.columns.values.tolist()
                features_to_remove = []
                for feature in features:
                    if feature not in x_features and feature not in y_feature:
                        features_to_remove.append(feature)
                dataframe_nostrings = FeatureIO(dataframe=dataframe_nostrings).remove_custom_features(features_to_remove=features_to_remove)
                dataframe_nostrings = self._perform_feature_selection(dataframe=dataframe_nostrings, x_features=x_features, y_feature=y_feature)
                x_features, y_feature = DataParser(configdict=self.configdict).get_features(dataframe=dataframe_nostrings, target_feature=y_feature)
            logging.debug("post-removal:%s" % dataframe_nostrings.columns)
            # Combine the input dataframe, which has undergone feature generation and normalization, with the grouped and labeled features of original dataframe
            # First, need to generate dataframe that only has the grouped and labeled features
            grouping_and_labeling_features = []
            duplicate_features = []
            if 'grouping_feature' in self.configdict['General Setup'].keys():
                grouping_and_labeling_features.append(grouping_feature)
            if 'labeling_features' in self.configdict['General Setup'].keys():
                for feature in labeling_features:
                    grouping_and_labeling_features.append(feature)
                    if feature in x_features:
                        if feature not in duplicate_features:
                            duplicate_features.append(feature)


            # Now merge dataframes
            dataframe_labeled_grouped = DataframeUtilities()._merge_dataframe_columns(dataframe1=dataframe_labeled, dataframe2=dataframe_grouped)
            dataframe_merged = DataframeUtilities()._merge_dataframe_columns(dataframe1=dataframe_nostrings, dataframe2=dataframe_labeled_grouped)

            # Need to remove duplicate features after merging.
            logging.debug("merged:%s" % dataframe_merged.columns)
            dataframe_rem = FeatureIO(dataframe=dataframe_merged).remove_duplicate_columns()

            myXdata, myydata, myx_features, myy_feature, dataframe_final = DataParser(configdict=self.configdict).parse_fromdataframe(dataframe=dataframe_rem, target_feature=y_feature)
            combined_x_features = list()
            logging.debug("total features:%s" % myx_features)
            for feature in myx_features:
                if (feature in original_x_features) or not(feature in original_columns): #originally designated, or created from feature generation
                    combined_x_features.append(feature)
            logging.debug("combined x features:%s" % combined_x_features)
            data_dict[data_name] = DataHandler(data = dataframe_final,
                                input_data = dataframe_final[combined_x_features],
                                target_data = myydata,
                                input_features = combined_x_features,
                                target_feature = myy_feature,
                                target_error_feature = target_error_feature,
                                labeling_features = labeling_features,
                                grouping_feature = grouping_feature) #
            logging.info('Parsed the input data located under %s' % data_path)
        return data_dict

    @timeit
    def _perform_feature_generation(self, dataframe):
        for k, v in self.configdict['Feature Generation'].items():
            # TODO: Here, True/False are strings. Change them with validator to be bools
            if k == 'add_magpie_features' and v == 'True':
                logging.info('FEATURE GENERATION: Adding Magpie features to your feature list')
                if self.configdict['Feature Generation']['include_magpie_atomic_features'] == 'True':
                    logging.info('FEATURE GENERATION: Include all atomic features: True')
                    mfg = MagpieFeatureGeneration(dataframe=dataframe, include_atomic_features=True)
                elif self.configdict['Feature Generation']['include_magpie_atomic_features'] == 'False':
                    logging.info('FEATURE GENERATION: Include all atomic features: False')
                    mfg = MagpieFeatureGeneration(dataframe=dataframe, include_atomic_features=False)
                else:
                    logging.info('FEATURE GENERATION: Include all atomic features: False')
                    mfg = MagpieFeatureGeneration(dataframe=dataframe, include_atomic_features=False)
                dataframe = mfg.generate_magpie_features(save_to_csv=True)
            if k == 'add_materialsproject_features' and v == 'True':
                logging.info('FEATURE GENERATION: Adding Materials Project features to your feature list')
                mpfg = MaterialsProjectFeatureGeneration(dataframe=dataframe, mapi_key=self.configdict['Feature Generation']['materialsproject_apikey'])
                dataframe = mpfg.generate_materialsproject_features(save_to_csv=True)
            #if k == 'add_citrine_features' and v is True:
            #    cfg = CitrineFeatureGeneration(dataframe=dataframe, api_key=self.configdict['Feature Generation']['citrine_apikey'])
            #    dataframe = cfg.generate_citrine_features()
        return dataframe

    @timeit
    def _perform_feature_selection(self, dataframe, x_features, y_feature):
        for k, v in self.configdict['Feature Selection'].items():
            # TODO: Here, True/False are strings. Change them with validator to be bools
            if k == 'remove_constant_features' and v == 'True':
                logging.info('FEATURE SELECTION: Removing constant features from your feature list')
                dr = DimensionalReduction(dataframe=dataframe, x_features=x_features, y_feature=y_feature)
                dataframe = dr.remove_constant_features()
                x_features, y_feature = DataParser(configdict=self.configdict).get_features(dataframe=dataframe, target_feature=y_feature)

            if k == 'feature_selection_algorithm':
                logging.info('FEATURE SELECTION: Selecting features using a %s algorithm' % v)
                fs = FeatureSelection(dataframe=dataframe, x_features=x_features, y_feature=y_feature)
                if v == 'forward':
                    if int(self.configdict['Feature Selection']['number_of_features_to_keep']) <= len(x_features):
                        dataframe = fs.forward_selection(number_features_to_keep=int(self.configdict['Feature Selection']['number_of_features_to_keep']), save_to_csv=True)
                    else:
                        logging.info('Warning: you have specified to keep more features than the total number of features in your dataset. Defaulting to keep all features in feature selection')
                        dataframe = fs.forward_selection(number_features_to_keep=int(len(x_features)), save_to_csv=True)
                if v == 'RFE':
                    if int(self.configdict['Feature Selection']['number_of_features_to_keep']) <= len(x_features):
                        dataframe = fs.recursive_feature_elimination(number_features_to_keep=int(self.configdict['Feature Selection']['number_of_features_to_keep']), save_to_csv=True)
                    else:
                        logging.info('Warning: you have specified to keep more features than the total number of features in your dataset. Defaulting to keep all features in feature selection')
                        dataframe = fs.recursive_feature_elimination(number_features_to_keep=int(len(x_features)), save_to_csv=True)
                if v == 'univariate':
                    if int(self.configdict['Feature Selection']['number_of_features_to_keep']) <= len(x_features):
                        dataframe = fs.univariate_feature_selection(number_features_to_keep=int(self.configdict['Feature Selection']['number_of_features_to_keep']), save_to_csv=True)
                    else:
                        logging.info('Warning: you have specified to keep more features than the total number of features in your dataset. Defaulting to keep all features in feature selection')
                        dataframe = fs.univariate_feature_selection(number_features_to_keep=int(len(x_features)), save_to_csv=True)
                if v == 'stability':
                    if int(self.configdict['Feature Selection']['number_of_features_to_keep']) <= len(x_features):
                        dataframe = fs.stability_selection(number_features_to_keep=int(self.configdict['Feature Selection']['number_of_features_to_keep']), save_to_csv=True)
                    else:
                        logging.info('Warning: you have specified to keep more features than the total number of features in your dataset. Defaulting to keep all features in feature selection')
                        dataframe = fs.stability_selection(number_features_to_keep=int(len(x_features)), save_to_csv=True)
        return dataframe

    def _gather_models(self):
        self.models_and_tests_setup = self.mastmlwrapper.process_config_keyword(keyword='Models and Tests to Run')
        model_list = []
        model_val = self.models_and_tests_setup['models']
        model_vals = list()
        if type(model_val) is str:
            logging.info('Getting model %s' % model_val)
            ml_model = self.mastmlwrapper.get_machinelearning_model(model_type=model_val)
            model_list.append(ml_model)
            logging.info('Adding model %s to queue...' % str(model_val))
            model_vals.append(model_val)
        elif type(model_val) is list:
            for model in model_val:
                logging.info('Getting model %s' % model)
                ml_model = self.mastmlwrapper.get_machinelearning_model(model_type=model)
                model_list.append(ml_model)
                logging.info('Adding model %s to queue...' % str(model))
                model_vals.append(model_val)
        return (model_list, model_val)

    def _gather_tests(self, mastmlwrapper, configdict, data_dict, model_list, save_path, model_vals):
        # Gather test types
        self.readme_html_tests.append("<H2>Tests</H2>\n")
        self.readme_html.append("<H2>Favorites</H2>\n")
        test_list = self.string_or_list_input_to_list(self.models_and_tests_setup['test_cases'])
        # Run the specified test cases for every model
        for test_type in test_list:
            logging.info('Looking up parameters for test type %s' % test_type)
            test_params = self.configdict["Test Parameters"][test_type]
            # Set data lists
            training_dataset_name_list = self.string_or_list_input_to_list(test_params['training_dataset'])
            training_dataset_list = list()
            for dname in training_dataset_name_list:
                training_dataset_list.append(self.data_dict[dname])
            test_params['training_dataset'] = training_dataset_list
            testing_dataset_name_list = self.string_or_list_input_to_list(test_params['testing_dataset'])
            testing_dataset_list = list()
            for dname in testing_dataset_name_list:
                testing_dataset_list.append(self.data_dict[dname])
            test_params['testing_dataset'] = testing_dataset_list
            # Run the test case for every model
            for midx, model in enumerate(self.model_list):
                # Set save path, allowing for multiple tests and models and potentially multiple of the same model (KernelRidge rbf kernel, KernelRidge linear kernel, etc.)
                test_folder = "%s_%s%i" % (test_type, model.__class__.__name__, midx)
                test_save_path = os.path.join(self.save_path, test_folder)
                self.test_save_paths.append(test_save_path)
                if not os.path.isdir(test_save_path):
                    os.mkdir(test_save_path)
                self.mastmlwrapper.get_machinelearning_test(test_type=test_type,
                        model=model, save_path=test_save_path, **test_params)
                logging.info('Ran test %s for your %s model' % (test_type, str(model)))
                self._update_models_and_data(test_folder, test_save_path, midx)
                self.readme_html.extend(self.make_links_for_favorites(test_folder, test_save_path))
                testrelpath = os.path.relpath(test_save_path, self.save_path)
                self.readme_html_tests.append('<A HREF="%s">%s</A><BR>\n' % (testrelpath, test_type))
        return test_list

    def _update_models_and_data(self, test_folder, test_save_path, model_index=None):
        test_short = test_folder.split("_")[0]
        if not (test_short in self.param_optimizing_tests): #no need
            logging.info("No parameter or data updates necessary.")
            return
        logging.info("UPDATING PARAMETERS from %s" % test_folder)
        param_dict = self._get_param_dict(os.path.join(test_save_path,"OPTIMIZED_PARAMS"))
        model_val = self.model_vals[model_index]
        self.mastmlwrapper.configdict["Model Parameters"][model_val].update(param_dict["model"])
        logging.info("New %s params: %s" % (model_val, self.mastmlwrapper.configdict["Model Parameters"][model_val]))
        self.model_list[model_index] = self.mastmlwrapper.get_machinelearning_model(model_type=model_val) #update model list IN PLACE
        logging.info("Updated model.")
        afm_dict = self._get_afm_args(os.path.join(test_save_path,"ADDITIONAL_FEATURES"))
        self._update_data_dict(afm_dict, param_dict)
        return

    def _update_data_dict(self, afm_dict=dict(), param_dict=dict()):
        if len(afm_dict.keys()) == 0:
            logging.info("No data updates necessary.")
            return
        for dname in self.data_dict.keys():
            for afm in afm_dict.keys():
                afm_kwargs = dict(afm_dict[afm])
                (feature_name, feature_data) = cf_help.get_custom_feature_data(class_method_str = afm,
                    starting_dataframe = self.data_dict[dname].data,
                    param_dict = dict(param_dict[afm]),
                    addl_feature_method_kwargs = dict(afm_kwargs))
                self.data_dict[dname].add_feature(feature_name, feature_data)
                self.data_dict[dname].input_features.append(feature_name)
                self.data_dict[dname].set_up_data_from_features()
                logging.info("Updated dataset %s data and input features with new feature %s" % (dname,afm))
            newcsv = os.path.join(self.save_path, "updated_%s.csv" % dname)
            self.data_dict[dname].data.to_csv(newcsv)
            logging.info("Updated dataset printed to %s" % newcsv)
        return

    def _get_param_dict(self, fname):
        pdict=dict()
        with open(fname,'r') as pfile:
            flines = pfile.readlines()
        for fline in flines:
            fline = fline.strip()
            [gene, geneidx, genevalstr] = fline.split(";")
            if not gene in pdict.keys():
                pdict[gene] = dict()
            if not (gene == 'model'):
                geneidx = int(geneidx) #integer key to match ParamOptGA
            if geneidx in pdict[gene].keys():
                raise ValueError("Param file at %s returned two of the same paramter" % fname)
            geneval = float(genevalstr)
            pdict[gene][geneidx] = geneval
        return pdict

    def _get_afm_args(self, fname):
        adict=dict()
        with open(fname,'r') as afile:
            alines = afile.readlines()
        for aline in alines:
            aline = aline.strip()
            [af_method, af_arg, af_argval] = aline.split(";")
            if not af_method in adict.keys():
                adict[af_method] = dict()
            adict[af_method][af_arg] = af_argval
        return adict

    def _move_log_and_input_files(self):
        cwd = os.getcwd()
        if not(self.save_path == cwd):
            oldlog = os.path.join(self.save_path, "MASTMLlog.log")
            copylog = os.path.join(cwd, "MASTMLlog.log")
            if os.path.exists(oldlog):
                os.remove(oldlog)
            shutil.move(copylog, self.save_path)
            copyconfig = os.path.join(cwd, str(self.configfile))
            shutil.copy(copyconfig, self.save_path)
        return

    def set_favorites_dict(self):
        fdict = dict()
        fdict["SingleFit"] = ["single_fit.png"]
        fdict["SingleFitGrouped"] = ["per_group_info/per_group_info.png"]
        fdict["SingleFitPerGroup"] = ["per_group_fits_overlay/per_group_fits_overlay.png"]
        fdict["KFoldCV"] = ["best_worst_overlay.png"]
        fdict["LeaveOneOutCV"] = ["loo_results.png"]
        fdict["LeaveOutPercentCV"] = ["best_worst_overlay.png"]
        fdict["LeaveOutGroupCV"] = ["leave_out_group.png"]
        fdict["ParamOptGA"] = ["OPTIMIZED_PARAMS"]
        fdict["PredictionVsFeature"] = [] #not sure
        self.favorites_dict=dict(fdict)
        return

    def make_links_for_favorites(self, test_folder, test_save_path):
        linklist=list()
        linkloc  = ""
        linktext = ""
        linkline = ""
        test_short = test_folder.split("_")[0]
        if test_short in self.favorites_dict.keys():
            flist = self.favorites_dict[test_short]
            for fval in flist:
                linkloc = os.path.join(test_save_path, fval)
                linkloc = os.path.relpath(linkloc, self.save_path)
                testrelpath = os.path.relpath(test_save_path, self.save_path)
                linkline = '<A HREF="%s">%s</A> from test <A HREF="%s">%s</A><BR><BR>\n' % (linkloc, fval, testrelpath, test_folder)
                linklist.append(linkline)
                if '.png' in fval:
                    imline = '<A HREF="%s"><IMG SRC="%s" height=300 width=400></A><BR>\n' % (linkloc, linkloc)
                    linklist.append(imline)
                else:
                    txtline = '<EMBED SRC="%s" width=75%%><BR>\n' % (linkloc)
                    linklist.append(txtline)
                linklist.append("<BR>\n") 
        return linklist

if __name__ == '__main__':
    if len(sys.argv) > 1:
        mastml = MASTMLDriver(configfile=sys.argv[1])
        mastml.run_MASTML()
        logging.info('Your MASTML runs are complete!')
    else:
        print('Specify the name of your MASTML input file, such as "mastmlinput.conf", and run as "python AllTests.py mastmlinput.conf" ')


