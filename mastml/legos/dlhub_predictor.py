# Note that this is work in progress and some hard-coded values were used for initial examples only and will be
# removed (and generalized) in the future

from mastml.legos import feature_generators
import pandas as pd
import numpy as np
from sklearn.externals import joblib
import os

def get_input_columns(training_data_path, exclude_columns):
    # Load in training data and get input columns
    #try:
    #    df_train = pd.read_csv(training_data_path)
    #except:
    #    df_train = pd.read_excel(training_data_path)
    df_train = training_data_path
    input_columns = [col for col in df_train.columns.tolist() if col not in exclude_columns]
    return input_columns

def featurize_mastml(prediction_data, scaler_path, training_data_path, exclude_columns):
    '''
    prediction_data: a string, list of strings, or path to an excel file to read in compositions to predict
    file_path (str): file path to test data set to featurize
    composition_column_name (str): name of column in test data containing material compositions. Just assume it is 'composition'
    scaler: sklearn normalizer, e.g. StandardScaler() object, fit to the training data
    training_data_path (str): file path to training data set used in original model fit
    '''

    # Write featurizer that takes chemical formula of test materials (from file), constructs correct feature vector then reports predictions

    # TODO: make this variable input
    COMPOSITION_COLUMN_NAME = 'composition'

    if type(prediction_data) is str:
        if '.xlsx' in prediction_data:
            df_new = pd.read_excel(prediction_data, header=0)
            compositions = df_new[COMPOSITION_COLUMN_NAME].tolist()
        elif '.csv' in prediction_data:
            df_new = pd.read_csv(prediction_data, header=0)
            compositions = df_new[COMPOSITION_COLUMN_NAME].tolist()
        else:
            compositions = [prediction_data]
            df_new = pd.DataFrame().from_dict(data={COMPOSITION_COLUMN_NAME: compositions})
    elif type(prediction_data) is list:
        compositions = prediction_data
        df_new = pd.DataFrame().from_dict(data={COMPOSITION_COLUMN_NAME: compositions})
    else:
        raise TypeError('prediction_data must be a composition in the form of a string, list of strings, or .csv or .xlsx file path')

    # Also get the training data so can build MAGPIE list and see which are constant features
    #if type(training_data_path) is str:
    #    if '.xlsx' in training_data_path:
    #        df_train = pd.read_excel(training_data_path, header=0)
    #    elif '.csv' in training_data_path:
    #        df_train = pd.read_csv(training_data_path, header=0)
    #else:

    df_train = training_data_path

    # Generate and use magpie featurizer using mastml
    magpie = feature_generators.Magpie(composition_feature=COMPOSITION_COLUMN_NAME,
                                       feature_types=['composition_avg', 'arithmetic_avg', 'max', 'min', 'difference'])
    magpie.fit(df_new)
    df_new_featurized = magpie.transform(df_new)

    # df_train may have other columns with it. Just take the composition column to make features ???
    df_train = pd.DataFrame(df_train[COMPOSITION_COLUMN_NAME])
    magpie.fit(df_train)
    df_train_featurized = magpie.transform(df_train)

    # Remove generated columns which are constant for every data entry
    constant_cols = df_train_featurized.columns[df_train_featurized.nunique() <= 1].tolist()

    # Only keep columns which are not constant from the training data df
    #df_new_featurized = df_new_featurized[df_train_featurized_noconst.columns.tolist()]
    df_new_featurized_noconst = df_new_featurized.drop(columns=constant_cols)

    # Unpack the scaler from a .pkl file if needed. Probably a better way to do this.
    #if os.path.splitext(scaler_path)[1] == '.pkl':
    #    scaler = joblib.load(scaler_path)

    # Normalize full feature set
    df_new_featurized_normalized = pd.DataFrame(scaler_path.transform(df_new_featurized_noconst),
                                                columns=df_new_featurized_noconst.columns.tolist(),
                                                index=df_new_featurized_noconst.index)

    # Trim the normalized feature set to only include the features used in training
    input_columns = get_input_columns(training_data_path=training_data_path, exclude_columns=exclude_columns)
    df_new_featurized_normalized_trimmed = df_new_featurized_normalized[input_columns]

    # Join featurized, normalized dataframe with read-in dataframe containing material compositions
    #df_new_featurized_normalized_trimmed_withcomp = pd.concat([df_new_featurized_normalized_trimmed, df_new[COMPOSITION_COLUMN_NAME]], 1)

    X_test= np.array(df_new_featurized_normalized_trimmed)
    return compositions, X_test

def make_prediction(model, prediction_data, scaler_path, training_data_path, exclude_columns=['composition', 'band_gap']):
    """
    dlhub_servable : a DLHubClient servable model, used to call DLHub to use cloud resources to run model predictions
    compositions (list) : list of composition strings for data points to predict
    X_test (np array) : array of test X feature matrix
    """

    # Featurize the prediction data
    compositions, X_test = featurize_mastml(prediction_data, scaler_path, training_data_path, exclude_columns)

    y_pred_new = model.predict(X_test)
    pred_dict = dict()
    for comp, pred in zip(compositions, y_pred_new.tolist()):
        pred_dict[comp] = pred

    # Save new predictions to excel file in cwd
    df_pred = pd.DataFrame.from_dict(pred_dict, orient='index', columns=['Predicted value'])
    df_pred.to_excel(os.path.join(os.getcwd(),'new_material_predictions.xlsx'))
    return pred_dict

def run_dlhub_prediction(comp_list):
    # comp_list (list): list of strings of material compositions to featurize and predict

    # Note: this function is meant to run in a DLHub container that will have access to the following files:
    #  model.pkl : a trained sklearn model
    #  selected.csv : csv file containing training data
    #  preprocessor.pkl : a preprocessor from sklearn

    # For now, assume we are running from job made on Google Colab. Files stored at /content/filename
    # Load scaler:
    try:
        scaler_path = joblib.load('content/preprocessor.pkl')
    except FileNotFoundError:
        scaler_path = joblib.load('preprocessor.pkl')
    # Load model:
    try:
        model = joblib.load('content/model.pkl')
    except FileNotFoundError:
        model = joblib.load('model.pkl')
    # Prediction data comps:
    prediction_data = comp_list
    # Load training data:
    try:
        training_data_path = pd.read_csv('content/selected.csv')
    except FileNotFoundError:
        training_data_path = pd.read_csv('selected.csv')

    pred_dict = make_prediction(model, prediction_data, scaler_path, training_data_path, exclude_columns=['composition', 'band_gap'])
    return pred_dict