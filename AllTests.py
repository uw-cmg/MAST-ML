import configuration_parser
import importlib
import data_parser
import matplotlib
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
        logging.info('Successfully parsed your MASTML input file')

        mastmlwrapper = MASTMLWrapper(configdict=configdict)
        datasetup = mastmlwrapper.process_config_keyword(keyword='Data Setup')
        print(datasetup)
        models_and_tests_setup = mastmlwrapper.process_config_keyword(keyword='Models and Tests to Run')
        print(models_and_tests_setup)

        # Temporary call to data_parser for now, but will later deprecate
        data = data_parser.parse(datasetup['data_path'], datasetup['weights'])
        data.set_x_features(datasetup['X'])
        data.set_y_feature(datasetup['y'])
        save_path = os.path.abspath(datasetup['save_path'])
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
        logging.info('Parsed the input data located under %s' % str(datasetup['data_path']))

        # Run listed tests for each model
        model_list = []
        for k, v in models_and_tests_setup.items():
            if k == "models":
                if type(v) is str:
                    print('Getting model %s' % v)
                    ml_model = mastmlwrapper.get_machinelearning_model(keyword=v)
                    model_list.append(ml_model)
                if type(v) is list:
                    for model in models_and_tests_setup['models']:
                        print('Getting model %s' % model)
                        ml_model = mastmlwrapper.get_machinelearning_model(keyword=model)
                        model_list.append(ml_model)
            if k == "test_cases":
                # Run the specified test cases for every model
                for i, model in enumerate(model_list):
                    import FullFit as FullFit
                    import KFoldCV as KFoldCV
                    KFoldCV.execute(model=model, data=data, savepath=save_path)
                    FullFit.execute(model=model, data=data, savepath=save_path)



        for model in model_list:
            print(model)
        # Temporary call to FullFit.py until remade into classes by Tam


        if os.path.exists(datasetup['save_path']+"/"+'MASTMLlog.log'):
            os.remove(datasetup['save_path']+"/"+'MASTMLlog.log')
            shutil.move(cwd + "/" + 'MASTMLlog.log', datasetup['save_path'])
        else:
            shutil.move(cwd+"/"+'MASTMLlog.log', datasetup['save_path'])

        """
        # Instead of using this, use MASTMLWrapper routine to call each model
        model = importlib.import_module(model).get()

        print("running test {}".format(case_name))
        case_base = case_name.split("_")[0]
        case = importlib.import_module(case_base)  # TTM
        kwargs = config[case_name]
        # TTM save for later: change and make new paths
        # curdir = os.getcwd()
        save_path = os.path.abspath(save_path)
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
        # os.chdir(save_path)
        case.execute(model, data, save_path, lwr_data=lwr_data, **kwargs)
        # os.chdir(curdir)
        matplotlib.pyplot.close("all")
        """



if __name__ == '__main__':
    if len(sys.argv) > 1:
        mastml = MASTMLDriver(configfile=sys.argv[1])
        mastml.run_MASTML()
    else:
        print('Specify the name of your MASTML input file, such as "mastmlinput.conf", and run as "python AllTests.py mastmlinput.conf" ')




"""
parameter_names = ['model', 'data_path', 'save_path', 'Y', 'X', 'lwr_data_path', 'weights']

all_tests = config.get('AllTests', 'test_cases').split(',')

for case_name in all_tests:
    parameter_values = []
    for parameter in parameter_names:
        if parameter == 'weights':
            if config.has_option(case_name, parameter):
                parameter_values.append(config.getboolean(case_name, parameter))
            else:
                parameter_values.append(config.getboolean('AllTests', parameter))
        else:
            if config.has_option(case_name, parameter):
                parameter_values.append(config.get(case_name, parameter))
            else:
                parameter_values.append(config.get('AllTests', parameter))

    model, data_path, save_path, y_data, x_data, lwr_data_path, weights = parameter_values

    model = importlib.import_module(model).get()
    x_data = x_data.split(',')

    data = data_parser.parse(data_path, weights)
    data.set_x_features(x_data)
    data.set_y_feature(y_data)

    #TTM+2 remove filter
    #data.add_exclusive_filter("Temp (C)", '<>', 290)
    #data.overwrite_data_w_filtered_data()

    #TTM eventually want to get rid of lwr_data
    lwr_data = data_parser.parse(lwr_data_path)
    #TTM y_data never delta sigma; always set features
    #if not y_data == "delta sigma":
    lwr_data.set_x_features(x_data)
    lwr_data.set_y_feature(y_data)

    #TTM filter when create ancillary databases in DataImportAndExport
    #Do not filter here.
    if y_data == "CD_delta_sigma_y_MPa": #TTM change field name
        pass #TTM filter in DataImportAndExport instead
        data.add_exclusive_filter("Alloy",'=', 29)
        data.add_exclusive_filter("Alloy",'=', 8)
        data.add_exclusive_filter("Alloy", '=', 1)
        data.add_exclusive_filter("Alloy", '=', 2)
        data.add_exclusive_filter("Alloy", '=', 14)
        data.overwrite_data_w_filtered_data()

        lwr_data.add_exclusive_filter("Alloy",'=', 29)
        lwr_data.add_exclusive_filter("Alloy", '=', 14)
        lwr_data.overwrite_data_w_filtered_data()

    elif y_data == "EONY delta sigma":
        pass #TTM filter in DataImportAndExport instead
        data.add_exclusive_filter("Temp (C)", '=', 310)
        data.overwrite_data_w_filtered_data()

        lwr_data.add_exclusive_filter("Temp (C)", '=', 310)
        lwr_data.overwrite_data_w_filtered_data()

    #TTM allow multiple runs of the same case with different parameters:
    print("running test {}".format(case_name))
    case_base = case_name.split("_")[0]
    case = importlib.import_module(case_base) #TTM
    kwargs = config[case_name]
    #TTM save for later: change and make new paths
    #curdir = os.getcwd()
    save_path = os.path.abspath(save_path)
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    #os.chdir(save_path)
    case.execute(model, data, save_path, lwr_data = lwr_data, **kwargs)
    #os.chdir(curdir)
    matplotlib.pyplot.close("all")
"""
