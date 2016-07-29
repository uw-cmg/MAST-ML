import configuration_parser
import importlib
import data_parser
import matplotlib
import sys

if len(sys.argv) > 1:
    config = configuration_parser.parse(sys.argv[1])
else:
    config = configuration_parser.parse('default.conf')

parameter_names = ['model', 'data_path', 'save_path', 'Y', 'X', 'lwr_data_path']

all_tests = config.get('AllTests', 'test_cases').split(',')

for case_name in all_tests:
    parameter_values = []
    for parameter in parameter_names:
        if config.has_option(case_name, parameter):
            parameter_values.append(config.get(case_name, parameter))
        else:
            parameter_values.append(config.get('AllTests', parameter))
    model, data_path, save_path, y_data, x_data, lwr_data_path = parameter_values

    model = importlib.import_module(model).get()
    x_data = x_data.split(',')

    data = data_parser.parse(data_path)
    data.set_x_features(x_data)
    data.set_y_feature(y_data)

    lwr_data = data_parser.parse(lwr_data_path)
    if not y_data == "delta sigma":
        lwr_data.set_x_features(x_data)
        lwr_data.set_y_feature(y_data)

    if y_data == "CD delta sigma":
        data.add_exclusive_filter("Alloy",'=', 29)
        data.add_exclusive_filter("Alloy", '=', 14)
        data.overwrite_data_w_filtered_data()

        lwr_data.add_exclusive_filter("Alloy",'=', 29)
        lwr_data.add_exclusive_filter("Alloy", '=', 14)
        lwr_data.overwrite_data_w_filtered_data()

    elif y_data == "EONY delta sigma":
        data.add_exclusive_filter("Temp (C)", '=', 310)
        data.overwrite_data_w_filtered_data()

        lwr_data.add_exclusive_filter("Temp (C)", '=', 310)
        lwr_data.overwrite_data_w_filtered_data()

    print("running test {}".format(case_name))
    case = importlib.import_module(case_name)
    case.execute(model, data, save_path, lwr_data)
    matplotlib.pyplot.close("all")

