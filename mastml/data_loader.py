"""
Module for loading checking the input data file
"""

import pandas as pd

def load_data(file_path, input_features=None, target_feature=None):
    " Loads in csv from filename and ensures required columns are present. Returns dataframe. "

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

    # Ensure they are all present:
    for feature in required_features:
        if feature not in df.columns:
            raise Exception(f"Data file does not have column '{feature}'")

    X, y = df[input_features], df[target_feature]

    ## Held out data for prediction only
    #n = 3* len(X) // 4
    #X_held_out, X = X[n:], X[:n]
    #y_held_out, y = y[n:], y[:n]
    #df_held_out, df = df[n:], df[:n] # ??? uh
    X_held_out = None
    y_held_out = None

    df = df.drop(target_feature, axis=1)
    return df, X, y, (X_held_out, y_held_out)
