import configuration_parser
import importlib

config = configuration_parser.parse()

parameter_names = ['model', 'data_path', 'save_path', 'Y', 'X']


all_tests = config.get('AllTests', 'test_cases').split(',')

for case_name in all_tests:
    parameter_values = []
    for parameter in parameter_names:
        if config.has_option(case_name, parameter):
            parameter_values.append(config.get(case_name, parameter))
        else:
            parameter_values.append(config.get('AllTests', parameter))
    model, data_path, save_path, y_data, x_data = parameter_values

    model = importlib.import_module(model).get()
    x_data = x_data.split(',')

    print("running test {}".format(case_name))
    case = importlib.import_module(case_name)
    case.execute(model, data_path, save_path, Y=y_data)
