__author__ = 'Ryan Jacobs, Tam Mayeshiba'
__maintainer__ = 'Ryan Jacobs'
__version__ = '1.0'
__email__ = 'rjacobs3@wisc.edu'
__date__ = 'October 14th, 2017'

import sys
import os
import logging
import shutil
import time
import matplotlib
import importlib
import pandas as pd
from MASTMLInitializer import ModelTestConstructor, ConfigFileValidator
from DataOperations import DataParser, DataframeUtilities
from FeatureGeneration import MagpieFeatureGeneration, MaterialsProjectFeatureGeneration, CitrineFeatureGeneration
from FeatureOperations import FeatureNormalization, FeatureIO, MiscFeatureOperations
from FeatureSelection import FeatureSelection, DimensionalReduction, LearningCurve
from DataHandler import DataHandler
from SingleFit import timeit

class MASTMLDriver(object):
    """
    Class responsible for organizing and executing a MASTML workflow

    Args:
        configfile (MASTML configfile object) : a MASTML input file, as a configfile object

    Methods:
        run_MASTML : executes the MASTML workflow

    """
    def __init__(self, configfile):
        self.configfile = configfile
        # Set in code
        self.general_setup = None
        self.csv_setup = None
        self.data_setup = None
        self.models_and_tests_setup = None
        self.configdict = None
        self.modeltestconstructor = None
        self.save_path = None #top-level save path
        self.data_dict = dict() #Data dictionary
        self.model_list = list() #list of actual models
        self.model_vals = list() #list of model names, like "gkrr_model"
        self.param_optimizing_tests = ["ParamOptGA","ParamGridSearch"]
        self.readme_html = list()
        self.readme_html_tests = list()
        self.start_time = None
        self.favorites_dict = dict()
        self.test_save_paths = list()
        self.logfilename = None
        return

    def run_MASTML(self):
        # Begin MASTML session
        self._initialize_mastml_session()
        self._initialize_html()
        self._set_favorites_dict()

        # Parse MASTML input file
        self.modeltestconstructor, self.configdict, errors_present = self._generate_mastml_wrapper()
        self.data_setup = self.modeltestconstructor._process_config_keyword(keyword='Data Setup')

        # General setup
        self._perform_general_setup()

        # Perform CSV setup (optional)
        if "CSV Setup" in self.configdict.keys():
            self._perform_csv_setup()

        # Get data name list
        data_name_list = list()
        for key in self.configdict['Data Setup'].keys():
            if key not in data_name_list:
                data_name_list.append(key)

        self.data_dict, y_feature = self._create_data_dict()

        # Gather models
        (self.model_list, self.model_vals) = self._gather_models(y_feature=y_feature)

        # Gather tests
        test_list, test_params = self._gather_tests()

        # End MASTML session
        self._move_log_and_input_files()
        self._end_html()

        return

    def _check_data_exists(self):
        if os.path.exists(self.configdict['Data Setup']['Initial']['data_path']):
            print('exists')
        else:
            print('gone')
        return

    def _split_csv_file(self):
        # Need dataframe and x and y features so can split CSV accordingly.
        data_path_list = []

        dataframe = DataParser(configdict=self.configdict).import_data(datapath=self.configdict['Data Setup']['Initial']['data_path'])
        y_feature = self.configdict['General Setup']['target_feature']
        other_features = []
        for column in dataframe.columns.values:
            if column not in y_feature:
                other_features.append(column)

        dataframe_x = dataframe.loc[:, other_features]
        count = 1
        for feature in y_feature:
            try:
                dataframe_y = dataframe.loc[:, feature]
            except KeyError:
                logging.info('Error detected: The feature names in the csv and input files do not match')
                print('The feature names in the csv and input files do not match. Please fix feature names and re-run MASTML')
                sys.exit()
            dataframe_new = DataframeUtilities.merge_dataframe_columns(dataframe1=dataframe_x, dataframe2=dataframe_y)
            # Write the new dataframe to new CSV, and update data_path_list
            data_path_split = os.path.split(self.configdict['Data Setup']['Initial']['data_path'])
            filename = data_path_split[1].split(".csv")
            data_path = data_path_split[0]+"/"+str(filename[0])+"_"+str(count)+".csv"
            dataframe_new.to_csv(data_path, index=False)
            data_path_list.append(data_path)
            count += 1

        # Last, add file data paths that are not part of original CSV file to split
        for key in self.configdict['Data Setup'].keys():
            if key != 'Initial':
                data_path_list.append(self.configdict['Data Setup'][key]['data_path'])

        return data_path_list

    def _initialize_mastml_session(self):
        level="INFO"
        envlevel = os.getenv("MASTML_LOGLEVEL")
        if not envlevel is None:
            level = envlevel
        testname = str(self.configfile).split('.')[0]
        self.logfilename = "MASTMLlog_" + str(testname) + ".log"
        logging.basicConfig(filename=self.logfilename, level=level)
        current_time = time.strftime('%Y'+'-'+'%m'+'-'+'%d'+', '+'%H'+' hours, '+'%M'+' minutes, '+'and '+'%S'+' seconds')
        logging.info('Initiated new MASTML session at: %s' % current_time)
        self.start_time = time.strftime("%Y-%m-%d, %H:%M:%S")
        logging.info("Using matplotlib backend %s" % matplotlib.get_backend())
        logging.info("Matplotlib defaults from %s" % matplotlib.matplotlib_fname())
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
        logpath = os.path.join(self.save_path, self.logfilename)
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
        modeltestconstructor = ModelTestConstructor(configdict=configdict)
        logging.info('Successfully read in and parsed your MASTML input file, %s' % str(self.configfile))
        return modeltestconstructor, configdict, errors_present

    def _perform_general_setup(self):
        self.general_setup = self.modeltestconstructor._process_config_keyword(keyword='General Setup')
        self.save_path = os.path.abspath(self.general_setup['save_path'])
        if os.path.exists(self.save_path):
            logging.info('Your specified save path already exists. Creating a new save path appended with the date/time of this MASTML run')
            current_time = time.strftime('%Y' + '-' + '%m' + '-' + '%d' + '-' + '%H' + '%M' + '%S')
            self.save_path = self.save_path+'_'+current_time
        if not os.path.isdir(self.save_path):
            os.mkdir(self.save_path)
        self.configdict['General Setup']['save_path'] = self.save_path
        return self.save_path

    def _perform_csv_setup(self):
        self.csv_setup = self.modeltestconstructor._process_config_keyword(keyword = "CSV Setup")
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
        return Xdata, ydata, x_features, y_feature, dataframe

    def _create_data_dict(self):
        data_dict=dict()
        for data_name in self.data_setup.keys():
            data_path = self.configdict['Data Setup'][data_name]['data_path']

            logging.info('Creating data dict for data path %s and data name %s' % (data_path, data_name))

            #data_weights = self.data_setup[data_name]['weights']
            if 'labeling_features' in self.general_setup.keys():
                labeling_features = self._string_or_list_input_to_list(self.general_setup['labeling_features'])
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
                if self.configdict['Feature Generation']['perform_feature_generation'] == True:
                    generate_features = True
                else:
                    generate_features = False
            else:
                generate_features = False

            if 'Feature Normalization' in self.configdict.keys():
                if self.configdict['Feature Normalization']['normalize_x_features'] == True:
                    normalize_x_features = True
                else:
                    normalize_x_features = False
                if self.configdict['Feature Normalization']['normalize_y_feature'] == True:
                    normalize_y_feature = True
                else:
                    normalize_y_feature = False
            else:
                normalize_x_features = False
                normalize_y_feature = False

            if 'Feature Selection' in self.configdict.keys():
                if self.configdict['Feature Selection']['perform_feature_selection'] == True:
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

            # Plot initial histogram of input target data
            DataframeUtilities().plot_dataframe_histogram(dataframe=dataframe[y_feature], title='Histogram of input data',
                                                          xlabel=y_feature, ylabel='Number of occurrences',
                                                          save_path=self.configdict['General Setup']['save_path'], file_name='input_data_histogram.png')

            original_x_features = list(x_features)
            original_columns = list(dataframe.columns)
            logging.debug("original columns: %s" % original_columns)
            # Remove any missing rows from dataframe
            #dataframe = dataframe.dropna()

            # Save off label and grouping data
            dataframe_labeled = pd.DataFrame()
            dataframe_grouped = pd.DataFrame()
            if not (labeling_features is None):
                dataframe_labeled = FeatureIO(dataframe=dataframe).keep_custom_features(features_to_keep=labeling_features, y_feature=y_feature)
                if normalize_x_features == True:
                    dataframe_labeled, scaler = FeatureNormalization(dataframe=dataframe_labeled, configdict=self.configdict).normalize_features(x_features=labeling_features, y_feature=y_feature)
            if not (grouping_feature is None):
                dataframe_grouped = FeatureIO(dataframe=dataframe).keep_custom_features(features_to_keep=[grouping_feature], y_feature=y_feature)

            # Generate additional descriptors, as specified in input file (optional)
            if generate_features:
                dataframe = self._perform_feature_generation(dataframe=dataframe)
                # Actually, the x_features_NOUSE is required if starting from no features and doing feature generation. Not renaming for now. RJ 7/17
                Xdata, ydata, x_features_NOUSE, y_feature, dataframe = DataParser(configdict=self.configdict).parse_fromdataframe(dataframe=dataframe, target_feature=y_feature)

            else:
                Xdata, ydata, x_features, y_feature, dataframe = DataParser(configdict=self.configdict).parse_fromdataframe(dataframe=dataframe, target_feature=y_feature)

            # First remove features containing strings before doing feature normalization or other operations, but don't remove grouping features
            if generate_features == True:
                nonstring_x_features, dataframe_nostrings = MiscFeatureOperations(configdict=self.configdict).remove_features_containing_strings(dataframe=dataframe, x_features=x_features_NOUSE)
                #Remove columns containing all entries of NaN
                dataframe_nostrings = dataframe_nostrings.dropna(axis=1, how='all')
                # Re-obtain x_feature list as some features may have been dropped
                Xdata, ydata, x_features_NOUSE, y_feature, dataframe_nostrings = DataParser(configdict=self.configdict).parse_fromdataframe(dataframe=dataframe_nostrings,target_feature=y_feature)
            else:
                nonstring_x_features, dataframe_nostrings = MiscFeatureOperations(configdict=self.configdict).remove_features_containing_strings(dataframe=dataframe, x_features=x_features)

            # Remove columns containing all entries of NaN
            dataframe_nostrings = dataframe_nostrings.dropna(axis=1, how='all')

            # Remove rows containing a single NaN value. This will handle removing a row with missing y-data, for example
            dataframe_nostrings = dataframe_nostrings.dropna(axis=0, how='any')

            # Re-obtain x_feature list as some features may have been dropped
            Xdata, ydata, x_features, y_feature, dataframe_nostrings = DataParser(configdict=self.configdict).parse_fromdataframe(dataframe=dataframe_nostrings, target_feature=y_feature)

            logging.debug("pre-changes:%s" % dataframe_nostrings.columns)

            # Normalize features (optional)
            if normalize_x_features == True or normalize_y_feature == True:
                fn = FeatureNormalization(dataframe=dataframe_nostrings, configdict=self.configdict)
                feature_normalization_type = self.configdict['Feature Normalization']['feature_normalization_type']
                feature_scale_min = self.configdict['Feature Normalization']['feature_scale_min']
                feature_scale_max = self.configdict['Feature Normalization']['feature_scale_max']
                dataframe_nostrings, scaler = fn.normalize_features(x_features=x_features, y_feature=y_feature, normalize_x_features=normalize_x_features, normalize_y_feature=normalize_y_feature, feature_normalization_type=feature_normalization_type, feature_scale_min=feature_scale_min, feature_scale_max=feature_scale_max)
                x_features, y_feature = DataParser(configdict=self.configdict).get_features(dataframe=dataframe_nostrings, target_feature=y_feature)

            # Perform feature selection and dimensional reduction, as specified in the input file (optional)
            if (select_features == True) and (y_feature in dataframe_nostrings.columns):
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
            dataframe_labeled_grouped = DataframeUtilities().merge_dataframe_columns(dataframe1=dataframe_labeled, dataframe2=dataframe_grouped)
            dataframe_merged = DataframeUtilities().merge_dataframe_columns(dataframe1=dataframe_nostrings, dataframe2=dataframe_labeled_grouped)

            #Add string columns back in
            string_x_features = list()
            for my_x_feature in x_features:
                if my_x_feature in nonstring_x_features:
                    pass
                else:
                    string_x_features.append(my_x_feature)
            logging.debug("string features: %s" % string_x_features)
            for string_x_feature in string_x_features:
                dataframe_merged[string_x_feature] = dataframe_orig_dropped_na[string_x_feature]

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

            # Get dataframe stats
            DataframeUtilities.save_all_dataframe_statistics(dataframe=dataframe_final, configdict=self.configdict)

        return data_dict, y_feature

    @timeit
    def _perform_feature_generation(self, dataframe):
        for k, v in self.configdict['Feature Generation'].items():
            if k == 'add_magpie_features' and v == True:
                logging.info('FEATURE GENERATION: Adding Magpie features to your feature list')
                mfg = MagpieFeatureGeneration(configdict=self.configdict, dataframe=dataframe)
                dataframe = mfg.generate_magpie_features(save_to_csv=True)
            if k == 'add_materialsproject_features' and v == True:
                logging.info('FEATURE GENERATION: Adding Materials Project features to your feature list')
                mpfg = MaterialsProjectFeatureGeneration(configdict=self.configdict, dataframe=dataframe, mapi_key=self.configdict['Feature Generation']['materialsproject_apikey'])
                dataframe = mpfg.generate_materialsproject_features(save_to_csv=True)
            if k == 'add_citrine_features' and v == True:
                logging.info('FEATURE GENERATION: Adding Citrine features to your feature list')
                cfg = CitrineFeatureGeneration(configdict=self.configdict, dataframe=dataframe, api_key=self.configdict['Feature Generation']['citrine_apikey'])
                dataframe = cfg.generate_citrine_features(save_to_csv=True)
        return dataframe

    @timeit
    def _perform_feature_selection(self, dataframe, x_features, y_feature):
        #for k, v in self.configdict['Feature Selection'].items():
            #if k == 'remove_constant_features' and v == True:
        if self.configdict['Feature Selection']['remove_constant_features'] == True:
            logging.info('FEATURE SELECTION: Removing constant features from your feature list')
            dr = DimensionalReduction(dataframe=dataframe, x_features=x_features, y_feature=y_feature)
            dataframe = dr.remove_constant_features()
            x_features, y_feature = DataParser(configdict=self.configdict).get_features(dataframe=dataframe, target_feature=y_feature)
        #for v in self.configdict['Feature Selection']['feature_selection_algorithm'].values():
            #if k == 'feature_selection_algorithm':
        logging.info('FEATURE SELECTION: Selecting features using a %s algorithm' % str(self.configdict['Feature Selection']['feature_selection_algorithm']))
        model_to_use = str(self.configdict['Feature Selection']['model_to_use_for_learning_curve'])
        fs = FeatureSelection(configdict=self.configdict, dataframe=dataframe, x_features=x_features, y_feature=y_feature, model_type=model_to_use)
        if self.configdict['Feature Selection']['feature_selection_algorithm'] == 'basic_forward_selection':
            dataframe = fs.feature_selection(feature_selection_type='basic_forward_selection',
                                             number_features_to_keep=self.configdict['Feature Selection']['number_of_features_to_keep'],
                                             use_mutual_info=self.configdict['Feature Selection']['use_mutual_information'])
        if self.configdict['Feature Selection']['feature_selection_algorithm'] == 'sequential_forward_selection':
            if int(self.configdict['Feature Selection']['number_of_features_to_keep']) <= len(x_features):
                dataframe = fs.sequential_forward_selection(number_features_to_keep=int(self.configdict['Feature Selection']['number_of_features_to_keep']))
            else:
                logging.info('Warning: you have specified to keep more features than the total number of features in your dataset. Defaulting to keep all features in feature selection')
                dataframe = fs.sequential_forward_selection(number_features_to_keep=int(len(x_features)))
        if self.configdict['Feature Selection']['feature_selection_algorithm'] == 'recursive_feature_elimination':
            if int(self.configdict['Feature Selection']['number_of_features_to_keep']) <= len(x_features):
                dataframe = fs.feature_selection(feature_selection_type = 'recursive_feature_elimination',
                                                   number_features_to_keep=int(self.configdict['Feature Selection']['number_of_features_to_keep']),
                                                   use_mutual_info=self.configdict['Feature Selection']['use_mutual_information'])
            else:
                logging.info('Warning: you have specified to keep more features than the total number of features in your dataset. Defaulting to keep all features in feature selection')
                dataframe = fs.feature_selection(feature_selection_type = 'recursive_feature_elimination',
                                                   number_features_to_keep=int(self.configdict['Feature Selection']['number_of_features_to_keep']),
                                                   use_mutual_info=self.configdict['Feature Selection']['use_mutual_information'])
            if self.configdict['Feature Selection']['generate_feature_learning_curve'] == True:
                learningcurve = LearningCurve(configdict=self.configdict, dataframe=dataframe, model_type=model_to_use)
                logging.info('Generating a feature learning curve using a %s algorithm' % str(self.configdict['Feature Selection']['feature_selection_algorithm']))
                learningcurve.generate_feature_learning_curve(feature_selection_algorithm='recursive_feature_elimination')
        if self.configdict['Feature Selection']['feature_selection_algorithm'] == 'univariate_feature_selection':
            if int(self.configdict['Feature Selection']['number_of_features_to_keep']) <= len(x_features):
                dataframe = fs.feature_selection(feature_selection_type='univariate_feature_selection',
                                                 number_features_to_keep=int(self.configdict['Feature Selection']['number_of_features_to_keep']),
                                                 use_mutual_info=self.configdict['Feature Selection']['use_mutual_information'])
            else:
                logging.info('Warning: you have specified to keep more features than the total number of features in your dataset. Defaulting to keep all features in feature selection')
                dataframe = fs.feature_selection(feature_selection_type='univariate_feature_selection',
                                                 number_features_to_keep=int(len(x_features)),
                                                 use_mutual_info=self.configdict['Feature Selection']['use_mutual_information'])
            if self.configdict['Feature Selection']['generate_feature_learning_curve'] == True:
                learningcurve = LearningCurve(configdict=self.configdict, dataframe=dataframe, model_type=model_to_use)
                logging.info('Generating a feature learning curve using a %s algorithm' % str(self.configdict['Feature Selection']['feature_selection_algorithm']))
                learningcurve.generate_feature_learning_curve(feature_selection_algorithm='univariate_feature_selection')
        return dataframe

    def _gather_models(self, y_feature):
        self.models_and_tests_setup = self.modeltestconstructor._process_config_keyword(keyword='Models and Tests to Run')
        model_list = []
        model_val = self.models_and_tests_setup['models']
        model_vals = list()

        if type(model_val) is str:
            logging.info('Getting model %s' % model_val)
            ml_model = self.modeltestconstructor.get_machinelearning_model(model_type=model_val, y_feature=y_feature)
            model_list.append(ml_model)
            logging.info('Adding model %s to queue...' % str(model_val))
            model_vals.append(model_val)
        elif type(model_val) is list:
            for model in model_val:
                logging.info('Getting model %s' % model)
                ml_model = self.modeltestconstructor.get_machinelearning_model(model_type=model, y_feature=y_feature)
                model_list.append(ml_model)
                logging.info('Adding model %s to queue...' % str(model))
                model_vals.append(model_val)
        return (model_list, model_val)

    def _gather_tests(self):
        # Gather test types
        self.readme_html_tests.append("<H2>Tests</H2>\n")
        self.readme_html.append("<H2>Favorites</H2>\n")
        test_list = self._string_or_list_input_to_list(self.models_and_tests_setup['test_cases'])
        # Run the specified test cases for every model
        for test_type in test_list:
            # Need to renew configdict each loop other issue with dictionary indexing occurs when you have multiple y_features
            #configdict = ConfigFileParser(configfile=sys.argv[1]).get_config_dict(path_to_file=os.getcwd())
            logging.info('Looking up parameters for test type %s' % test_type)
            test_params = self.configdict["Test Parameters"][test_type]

            # Set data lists
            training_dataset_name_list = self._string_or_list_input_to_list(test_params['training_dataset'])
            training_dataset_list = list()
            for dname in training_dataset_name_list:
                training_dataset_list.append(self.data_dict[dname])
            test_params['training_dataset'] = training_dataset_list

            testing_dataset_name_list = self._string_or_list_input_to_list(test_params['testing_dataset'])
            testing_dataset_list = list()
            for dname in testing_dataset_name_list:
                testing_dataset_list.append(self.data_dict[dname])
            test_params['testing_dataset'] = testing_dataset_list

            logging.info('on test %s' % test_type)

            # Run the test case for every model
            for midx, model in enumerate(self.model_list):
                if model is not None:
                    # Get name of target_feature for use in test_folder naming
                    for key, value in self.data_dict.items():
                        target_feature = value.target_feature
                    # Set save path, allowing for multiple tests and models and potentially multiple of the same model (KernelRidge rbf kernel, KernelRidge linear kernel, etc.)
                    test_folder = "%s_%s%i_%s" % (test_type, model.__class__.__name__, midx, str(target_feature))
                    test_save_path = os.path.join(self.save_path, test_folder)
                    self.test_save_paths.append(test_save_path)
                    if not os.path.isdir(test_save_path):
                        os.mkdir(test_save_path)

                    #print('passing model', model, 'for test type', test_type)

                    self.modeltestconstructor.get_machinelearning_test(test_type=test_type,
                                                                model=model, save_path=test_save_path, **test_params)
                    logging.info('Ran test %s for your %s model' % (test_type, str(model)))
                    self.readme_html.extend(self._make_links_for_favorites(test_folder, test_save_path))
                    testrelpath = os.path.relpath(test_save_path, self.save_path)
                    self.readme_html_tests.append('<A HREF="%s">%s</A><BR>\n' % (testrelpath, test_type))
                    test_short = test_folder.split("_")[0]
                    if test_short in self.param_optimizing_tests:
                        popt_warn = list()
                        popt_warn.append("Last test was a parameter-optimizing test.")
                        popt_warn.append("The test results may have specified an update of model parameters or data.")
                        popt_warn.append("Please run a separate test.conf file with updated data and model parameters.")
                        popt_warn.append("No further tests will be run on this file.")
                        for popt_warn_line in popt_warn:
                            logging.warning(popt_warn_line)
                            print(popt_warn_line)
                        break

        return test_list, test_params

    def _move_log_and_input_files(self):
        cwd = os.getcwd()
        logging.info('Your MASTML runs have completed successfully! Wrapping up MASTML session...')
        if self.save_path != cwd:
            log_old_location = os.path.join(cwd, self.logfilename)

            inputdata_name = self.configdict['Data Setup']['Initial']['data_path'].split('/')[-1]
            data_old_location = os.path.join(cwd, inputdata_name)

            shutil.copy(self.logfilename, self.save_path)
            if os.path.exists(log_old_location):
                os.remove(log_old_location)

            if os.path.exists(data_old_location):
                shutil.copy(data_old_location, self.save_path)

            copyconfig = os.path.join(cwd, str(self.configfile))
            shutil.copy(copyconfig, self.save_path)
        return

    def _set_favorites_dict(self):
        fdict = dict()
        fdict["SingleFit"] = ["single_fit.png"]
        fdict["SingleFitGrouped"] = ["per_group_info/per_group_info.png"]
        fdict["SingleFitPerGroup"] = ["per_group_fits_overlay/per_group_fits_overlay.png"]
        fdict["KFoldCV"] = ["best_worst_overlay.png"]
        fdict["LeaveOneOutCV"] = ["loo_results.png"]
        fdict["LeaveOutPercentCV"] = ["best_worst_overlay.png"]
        fdict["LeaveOutGroupCV"] = ["leave_out_group.png"]
        fdict["ParamOptGA"] = ["OPTIMIZED_PARAMS"]
        fdict["ParamGridSearch"] = ["OPTIMIZED_PARAMS","rmse_heatmap.png","rmse_heatmap_3d.png"]
        fdict["PredictionVsFeature"] = [] #not sure
        self.favorites_dict=dict(fdict)
        return

    def _make_links_for_favorites(self, test_folder, test_save_path):
        linklist=list()
        linkloc  = ""
        linktext = ""
        linkline = ""
        test_short = test_folder.split("_")[0]
        if test_short in self.favorites_dict.keys():
            flist = self.favorites_dict[test_short]
            for fval in flist:
                linkloc = os.path.join(test_save_path, fval)
                linklocrel = os.path.relpath(linkloc, self.save_path)
                testrelpath = os.path.relpath(test_save_path, self.save_path)
                linkline = '<A HREF="%s">%s</A> from test <A HREF="%s">%s</A><BR><BR>\n' % (linklocrel, fval, testrelpath, test_folder)
                linklist.append(linkline)
                if not (os.path.exists(linkloc)):
                    linklist.append("File not found.\n")
                else:
                    if '.png' in fval:
                        imline = '<A HREF="%s"><IMG SRC="%s" height=300 width=400></A><BR>\n' % (linklocrel, linklocrel)
                        linklist.append(imline)
                    else:
                        txtline = '<EMBED SRC="%s" width=75%%><BR>\n' % (linklocrel)
                        linklist.append(txtline)
                linklist.append("<BR>\n") 
        return linklist

    def _string_or_list_input_to_list(self, unknown_input_val):
        input_list = list()
        if type(unknown_input_val) is str:
            input_list.append(unknown_input_val)
        elif type(unknown_input_val) is list:
            for unknown_input in unknown_input_val:
                input_list.append(unknown_input)
        return input_list

if __name__ == '__main__':
    if len(sys.argv) > 1:
        mastml = MASTMLDriver(configfile=sys.argv[1])
        mastml.run_MASTML()
    else:
        logging.info('Specify the name of your MASTML input file, such as "mastmlinput.conf", and run as "python MASTML.py mastmlinput.conf" ')
        sys.exit()