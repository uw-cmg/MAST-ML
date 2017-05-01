__author__ = 'Ryan Jacobs'

import data_parser
import sys
import os
from MASTMLInitializer import ConfigFileParser, MASTMLWrapper
import logging
import shutil

class MASTMLDriver(object):

    def __init__(self, configfile):
        self.configfile = configfile
        logging.basicConfig(filename='MASTMLlog.log', level='INFO')

    def run_MASTML(self):
        cwd = os.getcwd()
        config = ConfigFileParser(configfile=self.configfile)
        logging.info('Successfully read in your MASTML input file, %s' % str(self.configfile))

        configdict = config.get_config_dict()
        mastmlwrapper = MASTMLWrapper(configdict=configdict)
        datasetup = mastmlwrapper.process_config_keyword(keyword='Data Setup')
        models_and_tests_setup = mastmlwrapper.process_config_keyword(keyword='Models and Tests to Run')
        logging.info('Successfully parsed your MASTML input file')

        # Temporary call to data_parser for now, but will later deprecate
        data = data_parser.parse(datasetup['data_path'], datasetup['weights'])
        data.set_x_features(datasetup['X'])
        data.set_y_feature(datasetup['y'])
        save_path = os.path.abspath(datasetup['save_path'])
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
        logging.info('Parsed the input data located under %s' % str(datasetup['data_path']))

        # Gather models and run listed tests for each model
        model_list = []
        for k, v in models_and_tests_setup.items():
            if k == "models":
                if type(v) is str:
                    print('Getting model %s' % v)
                    ml_model = mastmlwrapper.get_machinelearning_model(model_type=v)
                    model_list.append(ml_model)
                    logging.info('Adding model %s to queue...' % str(v))
                if type(v) is list:
                    for model in models_and_tests_setup['models']:
                        print('Getting model %s' % model)
                        ml_model = mastmlwrapper.get_machinelearning_model(model_type=model)
                        model_list.append(ml_model)
                        logging.info('Adding model %s to queue...' % str(model))
            if k == "test_cases":
                # Run the specified test cases for every model
                for i, model in enumerate(model_list):
                    mastmlwrapper.get_machinelearning_test(test_type=models_and_tests_setup['test_cases'][i],
                                                                     model=model, data=data, save_path=save_path)
                    logging.info('Ran test %s for your %s model' % (str(models_and_tests_setup['test_cases'][i]), str(model)))

        # Move input and log files to output directory, end MASTML session
        if not(os.path.abspath(datasetup['save_path']) == cwd):
            if os.path.exists(datasetup['save_path']+"/"+'MASTMLlog.log'):
                os.remove(datasetup['save_path']+"/"+'MASTMLlog.log')
            shutil.move(cwd+"/"+'MASTMLlog.log', datasetup['save_path'])
            shutil.copy(cwd+"/"+str(self.configfile), datasetup['save_path'])
        return

if __name__ == '__main__':
    if len(sys.argv) > 1:
        mastml = MASTMLDriver(configfile=sys.argv[1])
        mastml.run_MASTML()
        logging.info('Your MASTML runs are complete!')
    else:
        print('Specify the name of your MASTML input file, such as "mastmlinput.conf", and run as "python AllTests.py mastmlinput.conf" ')


