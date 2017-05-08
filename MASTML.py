__author__ = 'Ryan Jacobs, Tam Mayeshiba'

import data_parser
import sys
import os
from MASTMLInitializer import MASTMLWrapper, ConfigFileValidator
from DataParser import DataParser, FeatureIO
import logging
import shutil
import time

class MASTMLDriver(object):

    def __init__(self, configfile):
        self.configfile = configfile

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
        self._initalize_mastml_session()

        # Parse MASTML input file
        mastmlwrapper, configdict, errors_present = self._generate_mastml_wrapper()
        datasetup = mastmlwrapper.process_config_keyword(keyword='Data Setup')

        # General setup
        save_path = self._perform_general_setup(mastmlwrapper=mastmlwrapper)

        # Parse input data files
        Xdata, ydata, x_features, y_feature, dataframe, data_dict = self._parse_input_data(mastmlwrapper=mastmlwrapper, configdict=configdict)

        print(x_features)
        print(y_feature)

        #x_to_remove = ['x2', 'x3']
        fio = FeatureIO(dataframe=dataframe)
        #dataframe = ff.remove_custom_features(features_to_remove=x_to_remove)
        features_to_add = ['x4']
        data_to_add = dataframe['x1']
        dataframe = fio.add_custom_features(features_to_add=features_to_add, data_to_add=data_to_add)
        print(dataframe)
        Xdata, ydata, x_features, y_feature, dataframe = DataParser(configdict=configdict).parse_fromdataframe(dataframe=dataframe, target_feature='sin(x)', as_array=False)
        print(x_features)
        print(y_feature)
        #dataframe = ff.remove_duplicate_features()
        #print(dataframe)
        dataframe = fio.remove_duplicate_features_by_values(x_features=x_features, y_feature=y_feature)
        print(dataframe)

        """
        # Gather models
        model_list = self._gather_models(mastmlwrapper=mastmlwrapper)

        # Gather tests
        test_list = self._gather_tests(mastmlwrapper=mastmlwrapper, configdict=configdict, data_dict=data_dict,
                                       model_list=model_list, save_path=save_path)

        # End MASTML session
        self._move_log_and_input_files(mastmlwrapper=mastmlwrapper)
        """

        return

    def _initalize_mastml_session(self):
        logging.basicConfig(filename='MASTMLlog.log', level='INFO')
        current_time = time.strftime('%Y'+'-'+'%m'+'-'+'%d'+', '+'%H'+' hours, '+'%M'+' minutes, '+'and '+'%S'+' seconds')
        logging.info('Initiated new MASTML session at: %s' % current_time)
        return

    def _generate_mastml_wrapper(self):
        configdict, errors_present = ConfigFileValidator(configfile=self.configfile).run_config_validation()
        mastmlwrapper = MASTMLWrapper(configdict=configdict)
        logging.info('Successfully read in and parsed your MASTML input file, %s' % str(self.configfile))
        return mastmlwrapper, configdict, errors_present

    def _perform_general_setup(self, mastmlwrapper):
        generalsetup = mastmlwrapper.process_config_keyword(keyword='General Setup')
        save_path = os.path.abspath(generalsetup['save_path'])
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
        return save_path

    def _parse_input_data(self, mastmlwrapper, configdict):
        datasetup = mastmlwrapper.process_config_keyword(keyword='Data Setup')
        Xdata, ydata, x_features, y_feature, dataframe = DataParser(configdict=configdict).parse_fromfile(datapath=datasetup['Initial']['data_path'], as_array=False)

        # Tam's code is here
        data_dict=dict()
        datasetup = mastmlwrapper.process_config_keyword(keyword='Data Setup')
        for data_name in datasetup.keys():
            data_path = datasetup[data_name]['data_path']
            data_weights = datasetup[data_name]['weights']
            if not(os.path.isfile(data_path)):
                raise OSError("No file found at %s" % data_path)
            data_dict[data_name] = data_parser.parse(data_path, data_weights)
            #data_dict[data_name].set_x_features(datasetup['X']) #set in test classes, not here, since different tests could have different X and y features
            #data_dict[data_name].set_y_feature(datasetup['y'])
            logging.info('Parsed the input data located under %s' % data_path)


        return Xdata, ydata, x_features, y_feature, dataframe, data_dict

    def _gather_models(self, mastmlwrapper):
        models_and_tests_setup = mastmlwrapper.process_config_keyword(keyword='Models and Tests to Run')
        model_list = []
        model_val = models_and_tests_setup['models']
        #print(model_val)
        if type(model_val) is str:
            logging.info('Getting model %s' % model_val)
            ml_model = mastmlwrapper.get_machinelearning_model(model_type=model_val)
            model_list.append(ml_model)
            logging.info('Adding model %s to queue...' % str(model_val))
        elif type(model_val) is list:
            for model in model_val:
                logging.info('Getting model %s' % model)
                ml_model = mastmlwrapper.get_machinelearning_model(model_type=model)
                model_list.append(ml_model)
                logging.info('Adding model %s to queue...' % str(model))
        return model_list

    def _gather_tests(self, mastmlwrapper, configdict, data_dict, model_list, save_path):
        models_and_tests_setup = mastmlwrapper.process_config_keyword(keyword='Models and Tests to Run')
        generalsetup = mastmlwrapper.process_config_keyword(keyword='General Setup')
        # Gather test types
        test_list = self.string_or_list_input_to_list(models_and_tests_setup['test_cases'])
        # Run the specified test cases for every model
        for test_type in test_list:
            logging.info('Looking up parameters for test type %s' % test_type)
            test_params = configdict["Test Parameters"][test_type]
            # Set data lists
            training_dataset_name_list = self.string_or_list_input_to_list(test_params['training_dataset'])
            training_dataset_list = list()
            for dname in training_dataset_name_list:
                training_dataset_list.append(data_dict[dname])
            test_params['training_dataset'] = training_dataset_list
            testing_dataset_name_list = self.string_or_list_input_to_list(test_params['testing_dataset'])
            testing_dataset_list = list()
            for dname in testing_dataset_name_list:
                testing_dataset_list.append(data_dict[dname])
            test_params['testing_dataset'] = testing_dataset_list
            # Get default features if not set
            if not ('input_features' in test_params.keys()):
                test_params['input_features'] = self.string_or_list_input_to_list(generalsetup['input_features'])
            else:
                test_params['input_features'] = self.string_or_list_input_to_list(test_params['input_features'])
            if not ('target_feature' in test_params.keys()):
                test_params['target_feature'] = generalsetup['target_feature']
            if not ('target_error_feature' in test_params.keys()):
                if 'target_error_feature' in generalsetup.keys():
                    test_params['target_error_feature'] = generalsetup['target_error_feature']
            if 'labeling_features' in test_params.keys():
                test_params['labeling_features'] = self.string_or_list_input_to_list(test_params['labeling_features'])
            # Run the test case for every model
            for midx, model in enumerate(model_list):
                # Set save path, allowing for multiple tests and models and potentially multiple of the same model (KernelRidge rbf kernel, KernelRidge linear kernel, etc.)
                test_folder = "%s_%s%i" % (test_type, model.__class__.__name__, midx)
                test_save_path = os.path.join(save_path, test_folder)
                if not os.path.isdir(test_save_path):
                    os.mkdir(test_save_path)
                mastmlwrapper.get_machinelearning_test(test_type=test_type,
                                                       model=model, save_path=test_save_path,
                                                       **test_params)
                logging.info('Ran test %s for your %s model' % (test_type, str(model)))

        return test_list

    def _move_log_and_input_files(self, mastmlwrapper):
        cwd = os.getcwd()
        generalsetup = mastmlwrapper.process_config_keyword(keyword='General Setup')
        if not(os.path.abspath(generalsetup['save_path']) == cwd):
            if os.path.exists(generalsetup['save_path']+"/"+'MASTMLlog.log'):
                os.remove(generalsetup['save_path']+"/"+'MASTMLlog.log')
            shutil.move(cwd+"/"+'MASTMLlog.log', generalsetup['save_path'])
            shutil.copy(cwd+"/"+str(self.configfile), generalsetup['save_path'])
        return

if __name__ == '__main__':
    if len(sys.argv) > 1:
        mastml = MASTMLDriver(configfile=sys.argv[1])
        mastml.run_MASTML()
        logging.info('Your MASTML runs are complete!')
    else:
        print('Specify the name of your MASTML input file, such as "mastmlinput.conf", and run as "python AllTests.py mastmlinput.conf" ')


