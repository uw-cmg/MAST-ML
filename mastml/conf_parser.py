"""
Module for handling, parsing, and checking configuration files
"""

from distutils.util import strtobool


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
    feature_sections = ['FeatureGeneration', 'Clustering',  'FeatureNormalization', 'FeatureSelection']
    feature_section_dicts = [conf[name] for name in feature_sections if name in conf]
    all_sections = main_sections + feature_sections

    # Are all required sections present in the file?
    for name in main_sections:
        if name not in conf:
            conf[name] = dict()


    # If you don't have models, you don't take any input features
    if conf['Models'] == dict() and 'input_features' not in conf['GeneralSetup']:
        conf['GeneralSetup']['input_features'] = []


    # Are there any invalid sections?
    for section_name in conf:
        if section_name not in all_sections:
            raise Exception(f'[{section_name}] is not a valid section! Valid sections: {all_sections}')

    # Does a subsection-only section contain a parameter?
    dict_dicts = [conf, conf['DataSplits'], conf['Models']] + feature_section_dicts
    for dictionary in dict_dicts:
        for name, value in dictionary.items():
            if not isinstance(value, dict):
                raise TypeError(f"Parameter in subsection-only section: {name}={value}")

    # Collect all subsections which contain only parameters (no subsubsections):
    parameter_dicts = [conf['GeneralSetup']] + conf['Models'].values() + conf['DataSplits'].values()
    for feature_section in feature_section_dicts:
        parameter_dicts.extend(feature_section.values())

    # Do any parameter sections contain a subsection?
    # Also, cast the strings to their respective types
    for parameter_dict in parameter_dicts:
        for name, value in parameter_dict.items():
            if isinstance(value, dict):
                raise TypeError(f"Subsection in parameter-only section: {key}")
            # input_features/target_feature might have column named 'y' or 'off' or whatever, so don't fix it
            if name in ['input_features', 'target_feature']: continue
            parameter_dict[name] = fix_types(value)

    # Ensure all models are either classifiers or regressors: (raises error if mixed)
    is_classification = conf['is_classification'] = check_models_mixed(key.split('_')[0] for key in conf['Models'])

    ## Assign default values to unspecified or 'Auto' options: ##

    if conf['DataSplits'] == dict():
        conf['DataSplits']['NoSplit'] = dict()

    for name in feature_sections:
        if name not in conf or conf[name] == dict():
            if name == 'Clustering':
                conf[name] = dict()
                continue
            conf[name] = {'DoNothing': dict()}

    for name in ['input_features', 'target_feature']:
        if (name not in conf['GeneralSetup']) or (conf['GeneralSetup'][name] == 'Auto'):
            conf['GeneralSetup'][name] = None

    if 'metrics' in conf['GeneralSetup']:
        conf['metrics'] = conf['GeneralSetup']['metrics']
        del conf['GeneralSetup']['metrics']
    if 'metrics' not in conf or conf['metrics'] == 'Auto':
        if is_classification:
            conf['metrics'] = ['accuracy_score', 'precision_score', 'recall_score']
        else:
            conf['metrics'] = ['r2_score', 'root_mean_squared_error', 'mean_absolute_error', 'explained_variance_score']
    else: # User has specified their own specific metrics:
        metrics.check_names(conf['metrics'], is_classification)

    # TODO Grouping is not a real section, figure out how that would really work
    #if 'grouping_feature' in conf['Grouping']:
    #    conf['GeneralSetup']['grouping_feature'] = conf['Grouping']['grouping_feature']

    # TODO make a generic wrapper or indiviudla wrapper classes for these to import and use the
    # string for the score func

    _handle_selectors_references(conf['FeatureSelection'], is_classification)

    # Set the value of all subsections to be a pair of class,settings
    for dictionary in [conf['DataSplits'], conf['Models']] + [conf[name] for name in feature_sections]:
        for name, settings in dictionary.items():
            dictionary[name] = (name.split('_')[0], settings)

    return conf

def fix_types(maybe_list):
    " Takes user parameter string and gives true value "

    if isinstance(maybe_list, list):
        return [fix_types(item) for item in maybe_list]

    try: return strtobool(maybe_list)
    except ValueError: pass

    try: return int(maybe_list)
    except ValueError: pass

    try: return float(maybe_list)
    except ValueError: pass

    return str(maybe_list)

def _handle_selectors_references(selectors, is_classification):
    """
    Modifies each selector in `selectors` in place, turning strings referencing
    score_funcs and models into actual score_funcs and models
    """
    task = 'classification' if is_classification else 'regression'
    for selector_name, args_dict in selectors.items():
        selector_name = selector_name.split('_')[0]
        if selector_name in feature_selectors.score_func_selectors: # This selector requires a score func
            name_to_func = metrics.classification_score_funcs if is_classification else metrics.regression_score_funcs
            if 'score_func' in args_dict:
                try:
                    args_dict['score_func'] = name_to_func[args_dict['score_func']]
                except KeyError:
                    raise utils.InvalidValue(f"Score function '{args_dict['score_func']}' not valid for {task} tasks (inside feature selector {selector_name}). Valid score functions: name_to_func.keys()")
            else:
                args_dict['score_func'] = name_to_func['f_classif' if is_classification else 'f_regression']
