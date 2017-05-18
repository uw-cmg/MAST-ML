__author__ = 'Ryan Jacobs, Tam Mayeshiba'

import data_parser
import sys
import os
from MASTMLInitializer import MASTMLWrapper, ConfigFileValidator
from DataParser import DataParser, FeatureIO, FeatureNormalization
import logging
import shutil
import time
from custom_features import cf_help
from data_handling.DataHandler import DataHandler

class MASTMLDriver(object):

    def __init__(self, configfile):
        self.configfile = configfile
        #Set in code
        self.general_setup = None
        self.data_setup = None
        self.configdict = None
        self.mastmlwrapper = None
        self.save_path = None #top-level save path
        self.data_dict = dict() #Data dictionary
        self.model_list = list() #list of actual models
        self.model_vals = list() #list of model names, like "gkrr_model"
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

        # Parse MASTML input file
        self.mastmlwrapper, self.configdict, errors_present = self._generate_mastml_wrapper()
        self.data_setup = self.mastmlwrapper.process_config_keyword(keyword='Data Setup')

        # General setup
        self._perform_general_setup()

        # Parse input data files
        Xdata, ydata, x_features, y_feature, dataframe, self.data_dict = self._parse_input_data()

        print(x_features)
        print(y_feature)
        fn = FeatureNormalization(dataframe=dataframe)
        dataframe = fn.normalize_features(x_features=x_features, y_feature=y_feature)

        # Gather models
        (self.model_list, self.model_vals) = self._gather_models()

        # Gather tests
        test_list = self._gather_tests(mastmlwrapper=self.mastmlwrapper, configdict=self.configdict, data_dict=self.data_dict, model_list=self.model_list, save_path=self.save_path,
                model_vals = self.model_vals)

        # End MASTML session
        self._move_log_and_input_files()

        return

    def _initialize_mastml_session(self):
        logging.basicConfig(filename='MASTMLlog.log', level='INFO')
        current_time = time.strftime('%Y'+'-'+'%m'+'-'+'%d'+', '+'%H'+' hours, '+'%M'+' minutes, '+'and '+'%S'+' seconds')
        logging.info('Initiated new MASTML session at: %s' % current_time)
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

    def _parse_input_data(self):
        Xdata, ydata, x_features, y_feature, dataframe = DataParser(configdict=self.configdict).parse_fromfile(datapath=self.data_setup['Initial']['data_path'], as_array=False)
        # Tam's code is here
        data_dict=dict()
        for data_name in self.data_setup.keys():
            data_path = self.data_setup[data_name]['data_path']
            data_weights = self.data_setup[data_name]['weights']
            if not(os.path.isfile(data_path)):
                raise OSError("No file found at %s" % data_path)
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
            myXdata, myydata, myx_features, myy_feature, mydataframe = DataParser(configdict=self.configdict).parse_fromfile(datapath=self.data_setup[data_name]['data_path'], as_array=False)
            data_dict[data_name] = DataHandler(data = mydataframe, 
                                input_data = myXdata, 
                                target_data = myydata, 
                                input_features = myx_features,
                                target_feature = myy_feature,
                                target_error_feature = target_error_feature,
                                labeling_features = labeling_features,
                                grouping_feature = grouping_feature) #
            #data_dict[data_name] = data_parser.parse(data_path, data_weights)
            #data_dict[data_name].set_x_features(data_setup['X']) #set in test classes, not here, since different tests could have different X and y features
            #data_dict[data_name].set_y_feature(data_setup['y'])
            logging.info('Parsed the input data located under %s' % data_path)
        return Xdata, ydata, x_features, y_feature, dataframe, data_dict

    def _gather_models(self):
        models_and_tests_setup = self.mastmlwrapper.process_config_keyword(keyword='Models and Tests to Run')
        model_list = []
        model_val = models_and_tests_setup['models']
        model_vals = list()
        #print(model_val)
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
        models_and_tests_setup = mastmlwrapper.process_config_keyword(keyword='Models and Tests to Run')
        general_setup = mastmlwrapper.process_config_keyword(keyword='General Setup')
        # Gather test types
        test_list = self.string_or_list_input_to_list(models_and_tests_setup['test_cases'])
        param_optimizing_tests = ["ParamOptGA"]
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
                test_short = test_type.split("_")[0]
                if test_short in param_optimizing_tests:
                    logging.info("UPDATING PARAMETERS from %s" % test_type)
                    param_dict = self._get_param_dict(os.path.join(test_save_path,"OPTIMIZED_PARAMS"))
                    print(param_dict)
                    model_val = model_vals[midx]
                    mastmlwrapper.configdict["Model Parameters"][model_val].update(param_dict["model"])
                    print(mastmlwrapper.configdict["Model Parameters"][model_val])
                    model_list[midx] = mastmlwrapper.get_machinelearning_model(model_type=model_val)
                    logging.info("Updated model.")
                    afm_dict = self._get_afm_args(os.path.join(test_save_path,"ADDITIONAL_FEATURES"))
                    for dname in data_dict.keys():
                        for afm in afm_dict.keys():
                            afm_kwargs = dict(afm_dict[afm])
                            (feature_name, feature_data) = cf_help.get_custom_feature_data(class_method_str = afm,
                            starting_dataframe = data_dict[dname].data,
                            param_dict = dict(param_dict[afm]),
                            addl_feature_method_kwargs = dict(afm_kwargs))
                            data_dict[dname].add_feature(feature_name, feature_data)
                            data_dict[dname].input_features.append(feature_name)
                            data_dict[dname].set_up_data_from_features()
                            logging.info("Updated dataset %s data and input features with new feature %s" % (dname,afm))
        return test_list

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
        general_setup = self.mastmlwrapper.process_config_keyword(keyword='General Setup')
        if not(os.path.abspath(general_setup['save_path']) == cwd):
            if os.path.exists(general_setup['save_path']+"/"+'MASTMLlog.log'):
                os.remove(general_setup['save_path']+"/"+'MASTMLlog.log')
            shutil.move(cwd+"/"+'MASTMLlog.log', general_setup['save_path'])
            shutil.copy(cwd+"/"+str(self.configfile), general_setup['save_path'])
        return

if __name__ == '__main__':
    if len(sys.argv) > 1:
        mastml = MASTMLDriver(configfile=sys.argv[1])
        mastml.run_MASTML()
        logging.info('Your MASTML runs are complete!')
    else:
        print('Specify the name of your MASTML input file, such as "mastmlinput.conf", and run as "python AllTests.py mastmlinput.conf" ')


