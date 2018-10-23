"""
The data_loader module is used for importing data from user-specified csv or xlsx file to MAST-ML
"""

import pandas as pd
import logging
log = logging.getLogger('mastml')

def load_data(file_path, input_features=None, target_feature=None, grouping_feature = None, feature_blacklist=list()):
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
    if input_features is None and target_feature is None: # input is first n-1 and target is just n
        input_features = list(df.columns[:-1])
        target_feature = df.columns[-1]
    elif input_features is None: # input is all the features except the target feature
        input_features = [col for col in df.columns if col != target_feature]
    elif target_feature is None: # target is the last non-input feature
        for col in df.columns[::-1]:
            if col not in input_features:
                target_feature = col
                break

    # Collect required features:
    required_features = input_features + [target_feature]

    # Ensure they are all present:
    for feature in required_features:
        if feature not in df.columns:
            raise Exception(f"Data file does not have column '{feature}'")

    X, y = df[input_features], df[target_feature]

    log.info('blacklisted features, either from "not_input_features" or a "grouping_column":' +
                 str(feature_blacklist))
    # take blacklisted features out of X:
    X_noinput_dict = dict()
    for feature in set(feature_blacklist):
        X_noinput_dict[feature] = X[feature]
        X = X.drop(feature, axis=1)

    X_noinput = pd.DataFrame(X_noinput_dict)

    if grouping_feature:
        X_grouped = pd.DataFrame(df[grouping_feature])
    else:
        X_grouped = None

    df = df.drop(target_feature, axis=1)

    return df, X, X_noinput, X_grouped, y
