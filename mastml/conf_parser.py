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

    main_sections = ['GeneralSetup', 'DataSplits', 'Models']
    feature_sections = ['FeatureGeneration', 'Clustering',
                       'FeatureNormalization', 'FeatureSelection']
    feature_section_dicts = [conf[name] for name in feature_sections if name in conf]

    def verify_required_sections():
        for name in main_sections:
            if name not in conf:
                conf[name] = dict()
    verify_required_sections()

    def check_invalid_sections():
        all_sections = main_sections + feature_sections + ['PlotSettings']
        for section_name in conf:
            if section_name not in all_sections:
                raise Exception(f'[{section_name}] is not a valid section!'
                                f' Valid sections: {all_sections}')
    check_invalid_sections()

    def verify_subsection_only_sections():
        dict_dicts = [conf, conf['DataSplits'], conf['Models']] + feature_section_dicts
        for dictionary in dict_dicts:
            for name, value in dictionary.items():
                if not isinstance(value, dict):
                    raise TypeError(f"Parameter in subsection-only section: {name}={value}")
    verify_subsection_only_sections()

    # Collect all subsections which contain only parameters (no subsubsections):
    def parameter_dict_type_check_and_cast():
        parameter_dicts = conf['Models'].values() + conf['DataSplits'].values()
        for feature_section in feature_section_dicts:
            parameter_dicts.extend(feature_section.values())

        # Do any parameter sections contain a subsection?
        # Also, cast the strings to their respective types
        for parameter_dict in parameter_dicts:
            for name, value in parameter_dict.items():
                if isinstance(value, dict):
                    raise TypeError(f"Subsection in parameter-only section: {key}")
                # input_features and target_feature are always strings
                parameter_dict[name] = fix_types(value)
    parameter_dict_type_check_and_cast()

    # Ensure all models are either classifiers or regressors: (raises error if mixed)
    is_classification = conf['is_classification'] = check_models_mixed(
            key.split('_')[0] for key in conf['Models'])

    if conf['DataSplits'] == dict():
        conf['DataSplits']['NoSplit'] = dict()

    def make_empty_default_sections():
        for name in feature_sections:
            if name not in conf or conf[name] == dict():
                if name == 'Clustering':
                    conf[name] = dict()
                else:
                    conf[name] = {'DoNothing': dict()}
    make_empty_default_sections()

    GS = conf['GeneralSetup']

    def check_unknown_general_setup():
        all_settings =  ['input_features', 'target_feature', 'metrics',
                         'learning_curve_model', 'learning_curve_score']
        for name in GS:
            if name not in all_settings:
                raise utils.InvalidConfParameters(
                        f"[GeneralSetup] contains unknown setting {name}")
    check_unknown_general_setup()

    def set_default_features():
        for name in ['input_features', 'target_feature']:
            if (name not in GS) or (GS[name] == 'Auto'):
                GS[name] = None
    set_default_features()

    def verify_metrics():
        if 'metrics' not in GS or GS['metrics'] == 'Auto':
            if is_classification:
                GS['metrics'] = ['accuracy', 'precision_weighted', 'recall_weighted']
            else:
                GS['metrics'] = ['r2', 'root_mean_squared_error',
                                 'mean_absolute_error', 'explained_variance']
        GS['metrics'] = metrics.check_and_fetch_names(GS['metrics'], is_classification)
    verify_metrics()

    def change_selector_score_func_references():
        """
        Modifies each selector in `selectors` in place,
        turning strings referencing score_funcs into actual score_funcs
        """
        task = 'classification' if is_classification else 'regression'
        for selector_name, args_dict in conf['FeatureSelection'].items():
            selector_name = selector_name.split('_')[0]
            if selector_name not in feature_selectors.score_func_selectors: continue
            name_to_func = (metrics.classification_score_funcs if is_classification
                            else metrics.regression_score_funcs)
            if 'score_func' in args_dict:
                try:
                    args_dict['score_func'] = name_to_func[args_dict['score_func']]
                except KeyError:
                    raise utils.InvalidValue(
                            f"Score function '{args_dict['score_func']}' not valid for {task}"
                            f"tasks (inside feature selector {selector_name}). Valid score"
                            f"functions: name_to_func.keys()")
            else:
                args_dict['score_func'] = \
                        name_to_func['f_classif' if is_classification else 'f_regression']
    change_selector_score_func_references()

    def make_long_name_short_name_pairs():
        dictionaries = ([conf['DataSplits'], conf['Models']]
                        + [conf[name] for name in feature_sections])
        for dictionary in dictionaries:
            for name, settings in dictionary.items():
                dictionary[name] = (name.split('_')[0], settings)
    make_long_name_short_name_pairs()

    def check_and_boolify_plot_settings():
        default_false = ['feature_vs_target', 'data_learning_curve', 'feature_learning_curve']
        default_true  = ['target_histogram', 'main_plots', 'predicted_vs_true',
                         'predicted_vs_true_bars', 'best_worst_per_point']
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
        score_name = GS['learning_curve_score']
        d = metrics.check_and_fetch_names([score_name], is_classification)
        GS['learning_curve_score'] = make_scorer(d[score_name])
    if conf['PlotSettings']['data_learning_curve'] is True:
        check_learning_curve_settings()

    return conf

def fix_types(maybe_list):
    " Takes user parameter string and gives true value "

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
    if string.lower() == 'true':
        return True
    if string.lower() == 'false':
        return False
    raise ValueError
