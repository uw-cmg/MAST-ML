"""
Module for handling, parsing, and checking configuration files
"""

from sklearn.metrics import make_scorer
from configobj import ConfigObj
import logging

from . import metrics, utils
from .legos.model_finder import check_models_mixed
from .legos import feature_selectors, model_finder

log = logging.getLogger('mastml')

def parse_conf_file(filepath):
    "Accepts the filepath of a conf file and returns its parsed dictionary"

    conf = ConfigObj(filepath)

    main_sections = ['GeneralSetup', 'DataSplits', 'Models', 'LearningCurve']
    feature_sections = ['FeatureGeneration', 'Clustering',
                        'FeatureNormalization', 'FeatureSelection']
    feature_section_dicts = [conf[name] for name in feature_sections if name in conf]

    def set_required_sections_to_empty():
        for name in main_sections:
            if name not in conf:
                conf[name] = dict()
    set_required_sections_to_empty()

    def check_unknown_sections():
        all_sections = main_sections + feature_sections + ['PlotSettings']
        for section_name in conf:
            if section_name not in all_sections:
                raise Exception(f'[{section_name}] is not a valid section!'
                                f' Valid sections: {all_sections}')
    check_unknown_sections()

    def verify_subsection_only_sections():
        for dictionary in [conf, conf['DataSplits'], conf['Models']] + feature_section_dicts:
            for name, value in dictionary.items():
                if not isinstance(value, dict):
                    raise utils.InvalidConfSubSection(
                            f"Parameter in subsection-only section: {name}={value}")
    verify_subsection_only_sections()

    def parameter_dict_type_check_and_cast():
        parameter_dicts = conf['Models'].values() + conf['DataSplits'].values()
        for feature_section in feature_section_dicts:
            parameter_dicts.extend(feature_section.values())

        for parameter_dict in parameter_dicts:
            for name, value in parameter_dict.items():
                # Does this parameter-only section include a subsection?
                if isinstance(value, dict):
                    raise utils.InvalidConfSubSection(
                            f"Subsection in parameter-only section: {key}")
                # cast the strings to their respective types
                parameter_dict[name] = fix_types(value)
    parameter_dict_type_check_and_cast()

    # Ensure all models are either classifiers or regressors: (raises error if mixed)
    is_classification = conf['is_classification'] = check_models_mixed(
            key.split('_')[0] for key in conf['Models'])

    # Add the empty splitter if no splitters are specified:
    if conf['DataSplits'] == dict():
        conf['DataSplits']['NoSplit'] = dict()

    def set_unspecified_sections_to_empty_dict():
        for name in feature_sections:
            if name not in conf or conf[name] == dict():
                if name == 'Clustering':
                    conf[name] = dict()
                else:
                    conf[name] = {'DoNothing': dict()}
    set_unspecified_sections_to_empty_dict()

    GS = conf['GeneralSetup']

    def check_general_setup_settings_are_valid():
        all_settings =  ['input_features', 'target_feature', 'metrics',
                         'randomizer', 'validation_columns', 'not_input_features', 'grouping_feature']
        for name in GS:
            if name not in all_settings:
                raise utils.InvalidConfParameters(
                        f"[GeneralSetup] contains unknown setting {name}.\n"
                        f"Valid GeneralSetup options are: {all_settings}")
    check_general_setup_settings_are_valid()

    # Find grouping features and 'not_input_features' to blacklist out of X (see data loader)
    def collect_grouping_features():
        for section in conf:
            if not isinstance(conf[section], dict):
                continue
            for subsection in conf[section]:
                SS = conf[section][subsection]
                if not isinstance(SS, dict):
                    continue
                if 'grouping_column' in SS.keys():
                    logging.debug('found grouping_feature: ' + SS['grouping_column'])
                    yield SS['grouping_column']
    # Issue here where if clusters are automatically generated, new column is made but isn't in intitial df, even though
    # listed as grouping_feature. Here, just have to remember to put grouping_feature names in not_input_features
    #feature_blacklist = list(collect_grouping_features())
    feature_blacklist = list()
    # default not_input_features to a list
    if 'not_input_features' not in GS:
        GS['not_input_features'] = list()
    else:
        if type(GS['not_input_features']) is str:
            new_list = list()
            new_list.append(GS['not_input_features'])
            GS['not_input_features'] = new_list
        elif type(GS['not_input_features']) is list:
            pass

    # and add the discovered ones to the list
    GS['not_input_features'] += feature_blacklist
    #GS['not_input_features'] = [f for f in feature_blacklist if f not in GS['not_input_features']]

    def set_randomizer_setting():
        if 'randomizer' in GS:
            GS['randomizer'] = mybool(GS['randomizer'])
        else:
            GS['randomizer'] = False
    set_randomizer_setting()


    def set_default_features():
        for name in ['input_features', 'target_feature']:
            if (name not in GS) or (GS[name] == 'Auto'):
                GS[name] = None
    set_default_features()

    def set_default_metrics():
        if 'metrics' not in GS or GS['metrics'] == 'Auto':
            if is_classification:
                GS['metrics'] = ['accuracy', 'precision_binary', 'recall_binary', 'f1_binary']
            else:
                GS['metrics'] = ['R2', 'root_mean_squared_error',
                                 'mean_absolute_error', 'rmse_over_stdev']
    set_default_metrics()

    # Turn names of metrics into actual metrics:
    # If only one metric, map string to list for parsing
    if type(GS['metrics']) is str:
        el = GS['metrics']
        GS['metrics'] = list()
        GS['metrics'].append(el)

    GS['metrics'] = metrics.check_and_fetch_names(GS['metrics'], is_classification)

    def change_score_func_strings_into_actual_score_funcs():
        for selector_name, args_dict in conf['FeatureSelection'].items():
            class_name = selector_name.split('_')[0]
            if class_name not in feature_selectors.score_func_selectors: continue
            name_to_func = (metrics.classification_score_funcs
                            if is_classification
                            else metrics.regression_score_funcs)
            if 'score_func' in args_dict:
                try:
                    args_dict['score_func'] = name_to_func[args_dict['score_func']]
                except KeyError:
                    task = 'classification' if is_classification else 'regression'
                    raise utils.InvalidValue(
                            f"Score function '{args_dict['score_func']}' not valid for {task}"
                            f" tasks (inside feature selector {selector_name}). Valid score"
                            f" functions: {list(name_to_func.keys())}")
            else: # default to f_classif or f_regression
                args_dict['score_func'] = \
                        name_to_func['f_classif' if is_classification else 'f_regression']
    change_score_func_strings_into_actual_score_funcs()

    def make_long_name_short_name_pairs():
        dictionaries = ([conf['DataSplits'], conf['Models']]
                        + [conf[name] for name in feature_sections])
        for dictionary in dictionaries:
            for name, settings in dictionary.items():
                dictionary[name] = (name.split('_')[0], settings)
    make_long_name_short_name_pairs()

    def check_and_boolify_plot_settings():
        default_false = ['feature_vs_target']
        default_true = ['target_histogram', 'train_test_plots', 'predicted_vs_true',
                         'predicted_vs_true_bars', 'best_worst_per_point', 'average_normalized_errors',
                         'average_cumulative_normalized_errors']
        all_settings = default_false + default_true
        if 'PlotSettings' not in conf:
            conf['PlotSettings'] = dict()
        PS = conf['PlotSettings']
        for name, value in PS.items():
            if name not in all_settings:
                raise utils.InvalidConfParameters(f"[PlotSettings] parameter '{name}' is unknown")
            try:
                PS[name] = mybool(value)
            except ValueError:
                raise utils.InvalidConfParameters(
                    f"[PlotSettings] parameter '{name}' must be a boolean")
        for name in default_false:
            if name not in PS:
                PS[name] = False
        for name in default_true:
            if name not in PS:
                PS[name] = True
    check_and_boolify_plot_settings()

    def check_learning_curve_settings():
        if 'learning_curve_model' not in GS:
            raise utils.InvalidConfParameters("You enabled data_learning_curve plots but you did"
                                              "not specify learning_curve_model in [GeneralSetup]")
        if 'learning_curve_score' not in GS:
            raise utils.InvalidConfParameters("You enabled data_learning_curve plots but you did"
                                              "not specify learning_curve_score in [GeneralSetup]")
    if conf['LearningCurve']:
        score_name = conf['LearningCurve']['scoring']
        d = metrics.check_and_fetch_names([score_name], is_classification)
        greater_is_better, score_func = d[score_name]
        conf['LearningCurve']['scoring'] = make_scorer(score_func, greater_is_better=True)

    return conf

def fix_types(maybe_list):
    "Takes user parameter string and gives python value"

    if isinstance(maybe_list, list):
        return [fix_types(item) for item in maybe_list]

    try: return mybool(maybe_list)
    except ValueError: pass

    try: return int(maybe_list)
    except ValueError: pass

    try: return float(maybe_list)
    except ValueError: pass

    return str(maybe_list)

def mybool(string):
    "Turn string representing bool into actual bool"
    if string.lower() == 'true':
        return True
    if string.lower() == 'false':
        return False
    raise ValueError
