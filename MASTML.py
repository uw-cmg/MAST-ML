__author__ = 'Ryan Jacobs'

import data_parser
import sys
import os
from MASTMLInitializer import ConfigFileParser, MASTMLWrapper, ConfigFileValidator
import logging
import shutil
import time
from pprint import pprint

class MASTMLDriver(object):

    def __init__(self, configfile):
        self.configfile = configfile
        logging.basicConfig(filename='MASTMLlog.log', level='INFO')

    def run_MASTML(self):
        current_time = time.strftime('%Y'+'-'+'%m'+'-'+'%d'+', '+'%H'+' hours, '+'%M'+' minutes, '+'and '+'%S'+' seconds')
        logging.info('Initiated new MASTML session at: %s' % current_time)
        # Parse MASTML input file
        mastmlwrapper, configdict, errors_present = self._generate_mastml_wrapper()
        logging.info('Successfully read in and parsed your MASTML input file, %s' % str(self.configfile))

        generalsetup = mastmlwrapper.process_config_keyword(keyword='General Setup')
        datasetup = mastmlwrapper.process_config_keyword(keyword='Data Setup')

        # Parse input data files
        # Temporary call to data_parser for now, but will later deprecate
        data = data_parser.parse(datasetup['Initial']['data_path'], datasetup['Initial']['weights'])
        data.set_x_features(generalsetup['input_features'])
        data.set_y_feature(generalsetup['target_feature'])
        save_path = os.path.abspath(generalsetup['save_path'])
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
        logging.info('Parsed the input data located under %s' % str(datasetup['Initial']['data_path']))

        # Gather models
        model_list = self._gather_models(mastmlwrapper=mastmlwrapper)
        # Gather tests
        test_list = self._gather_tests(mastmlwrapper=mastmlwrapper, configdict=configdict, model_list=model_list,
                                       data=data, save_path=generalsetup['save_path'])
        # End MASTML session
        self._move_log_and_input_files(mastmlwrapper=mastmlwrapper)
        return

    def _generate_mastml_wrapper(self):
        configdict, errors_present = ConfigFileValidator(configfile=self.configfile).run_config_validation()
        mastmlwrapper = MASTMLWrapper(configdict=configdict)
        return mastmlwrapper, configdict, errors_present

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

    def _gather_tests(self, mastmlwrapper, configdict, model_list, data, save_path):
        models_and_tests_setup = mastmlwrapper.process_config_keyword(keyword='Models and Tests to Run')
        test_list = []
        test_val = models_and_tests_setup['test_cases']
        if type(test_val) is str:
            test_list.append(test_val)
        elif type(test_val) is list:
            for test in test_val:
                test_list.append(test)
        # Run the specified test cases for every model
        for test_type in test_list:
            logging.info('Looking up parameters for test type %s' % test_type)
            test_params = configdict["Test Parameters"][test_type]
            for midx, model in enumerate(model_list):
                mastmlwrapper.get_machinelearning_test(test_type=test_type,
                        model=model, data=data, save_path=save_path,
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


