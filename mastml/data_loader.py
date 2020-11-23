"""
The data_loader module is used for importing data from user-specified csv or xlsx file to MAST-ML
"""

import pandas as pd
import logging
from mastml import utils
log = logging.getLogger('mastml')

def load_data(file_path, input_features=None, input_target=None, input_grouping= None, feature_blacklist=list()):
    """
    Method that accepts the filepath of an input data file and returns a full dataframe and parsed X and y dataframes

    Args:
        file_path: (str), path to data file

        input_features: (str), column names to be used as input features (X data). If 'Auto', then takes all columns that are not
        listed in target_feature or feature_blacklist fields.

        target_feature: (str), column name for data to be fit to (y data).

        grouping_feature: (str), column names used to group data in user-defined grouping scheme

    Returns:
        df: (dataframe), full dataframe of the input X data (y data is removed)

        X: (dataframe), dataframe containing only the X data from the data file

        X_noinput: (dataframe), dataframe containing the columns of the original X data that are not used as input features

        X_grouped: (dataframe), dataframe containing the columns of hte original X data that correspond to a data grouping scheme

        y: (dataframe), dataframe containing only the y data from the data file

    """

    # Load data
    try:
        df = pd.read_csv(file_path)
    except:
        df = pd.read_excel(file_path)

    # Assign default values to input_features and target_feature;
    if input_features is None and input_target is None: # input is first n-1 and target is just n
        input_features = list(df.columns[:-1])
        target_feature = df.columns[-1]
    elif input_features is None: # input is all the features except the target feature
        input_features = [col for col in df.columns if col != input_target]
    elif input_target is None: # target is the last non-input feature
        for col in df.columns[::-1]:
            if col not in input_features:
                target_feature = col
                break

    # Collect required features:
    if type(input_features) is str:
        input_features = [input_features]
    required_features = input_features + [input_target]

    # Ensure they are all present:
    for feature in required_features:
        if feature not in df.columns:
            raise Exception(f"Data file does not have column '{feature}'")

    X, y = df[input_features], df[input_target]

    log.info('blacklisted features, either from "input_other" or a "input_grouping":' +
                 str(feature_blacklist))
    # take blacklisted features out of X:
    X_noinput_dict = dict()
    for feature in set(feature_blacklist):
        # If input_features = Auto, all included and blacklisted features need removal; if manual may not have all features
        if feature in X.columns:
            X_noinput_dict[feature] = X[feature]
            X = X.drop(feature, axis=1)
        else:
            log.info('Blacklisted feature ' + str(feature) + ' already not present in dataframe')

    # Need this block when input features not set to Auto
    for feature in set(feature_blacklist):
        if feature not in X_noinput_dict.keys():
            X_noinput_dict[feature] = df[feature]

    X_noinput = pd.DataFrame(X_noinput_dict)

    if input_grouping:
        X_grouped = pd.DataFrame(df[input_grouping])
    else:
        X_grouped = None

    df = df.drop(input_target, axis=1)

    #Check if features are unambiguously selected
    for feature in X_noinput.columns:
        if feature in X.columns:
            raise utils.ConfError('An error has occurred where the same feature in both the "input_features" and '
                                  '"input_other" fields. Please correct your input file and re-run MAST-ML')

    return df, X, X_noinput, X_grouped, y
