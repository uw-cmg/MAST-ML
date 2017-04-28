#import configuration_parser
import importlib
import data_parser
import matplotlib
import sys
import os
from ConfigFileParser import ConfigFileParser
from pprint import pprint
import logging


"""
if len(sys.argv) > 1:
    config = configuration_parser.parse(sys.argv[1])
else:
    config = configuration_parser.parse('default.conf')
"""

class MASTMLInitializer(object):

    def __init__(self, configfile):
        self.configfile = configfile
        logging.basicConfig(filename='MASTMLlog.log', level='INFO')

    def run_MASTML(self):
        config = ConfigFileParser(configfile=self.configfile)
        logging.info('Successfully read in your MASTML input file, %s' % str(self.configfile))
        configdict = config.get_config_dict()
        logging.info('Successfully parsed your MASTML input file')

        """
        print(configdict)
        configdict_depth = config._get_config_dict_depth(test_dict=configdict)
        print(configdict_depth)

        section_lvl1_list = []; section_lvl2_list = []
        for k, v in configdict.items():
            section_lvl1_list.append(k)
            if config._get_config_dict_depth(test_dict=v) > 1:
                for kk, vv in v.items():
                    section_lvl2_list.append(kk)

        print(section_lvl1_list)
        print(section_lvl2_list)
        """
        all_tests = configdict.get('Models and Tests to Run')
        print(all_tests)
        print(all_tests['model'])
        for case in all_tests['test_cases']:
            print(case)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        mastml = MASTMLInitializer(configfile=sys.argv[1])
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
