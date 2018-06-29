"""
Module for loading checking the input data file
"""

import pandas as pd

def load_data(file_path, input_features=None, target_feature=None, grouping_feature=None, clustering_features=None):
    " Loads in csv from filename and ensures required columns are present. Returns dataframe. "
    # TODO: modify this to omit the target feature from the input features by default if the target feature is supplied

    # Load data
    df = pd.read_csv(file_path)

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

    #if grouping_feature is not None:
    #    required_features.append(grouping_feature)
    #if clustering_features is not None:
    #    required_features.extend(clustering_features)

    # Ensure they are all present:
    for feature in required_features:
        if feature not in df.columns:
            raise Exception(f"Data file does not have column '{feature}'")

    df = df[required_features] # drop the columns we aren't using

    return df, input_features, target_feature
