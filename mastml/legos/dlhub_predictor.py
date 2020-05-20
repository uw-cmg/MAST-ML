# Note that this is work in progress and some hard-coded values were used for initial examples only and will be
# removed (and generalized) in the future

from mastml.legos import feature_generators
import pandas as pd
import numpy as np
from sklearn.externals import joblib
import glob

def get_input_columns(training_data_path, exclude_columns):
    # Load in training data and get input columns
    try:
        df_train = pd.read_csv(training_data_path)
    except:
        df_train = pd.read_excel(training_data_path)
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
    if type(training_data_path) is str:
        if '.xlsx' in training_data_path:
            df_train = pd.read_excel(training_data_path, header=0)
        elif '.csv' in training_data_path:
            df_train = pd.read_csv(training_data_path, header=0)

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
    # Run the predictions on the DLHub server
    #dl = DLHubClient()
    # Ryan Chard: it seems this needs to be changed to something like what is commented below:
    # model = joblib.load(servable['dlhub']['files']['model'])
    #y_pred_new = model.predict(X_test)
    #y_pred_new = dl.run(name=dlhub_servable, inputs=X_test.tolist())
    y_pred_new = model.predict(X_test)
    pred_dict = dict()
    for comp, pred in zip(compositions, y_pred_new.tolist()):
        pred_dict[comp] = pred

    # Save new predictions to excel file
    df_pred = pd.DataFrame.from_dict(pred_dict, orient='index', columns=['Predicted value'])
    df_pred.to_excel('new_material_predictions.xlsx')
    return pred_dict

def run_dlhub_prediction(comp_list):
    # dlhub_predictor_dict: dict containing the following two keys:
    #       dlhub_servable: the servable name. This is needed because it runs dlhub.run() internally to make the model inference.
    #                For this example, use 'rjacobs3_wisc/Bandgap_GW_2020_04_20'
    #       prediction_data: the material composition to be featurized and predicted. This is what we would like the new input
    #                   DLHubClient().run() to be, e.g. DLHubClient().run(dlhub_servable, prediction_data), which would
    #                   now be a composition (e.g. "Al2O3") instead of a featurized matrix of data, which would now be
    #                   done internally.
    # scaler_path: the .pkl preprocessor file. This is used in the featurization process. This is "StandardScaler.pkl"
    #              in the example servable
    # training_data_path: The training data used to train the saved model. Needed to get feature names to select in
    #                    featurization step. This is "selected.csv" in the example servable.
    # exclude_columns: Other column names that are in the "selected.csv" file but not used in featurization. Just hard
    #                   coded for now, will make general later if this works as expected

    # Note: this function is meant to run in a DLHub container that will have access to the following files:
    #  model.pkl : a trained sklearn model
    #  selected.csv : csv file containing training data
    #  preprocessor.pkl : a preprocessor from sklearn

    #dlhub_servable = dlhub_predictor_dict['dlhub_servable']
    #prediction_data = dlhub_predictor_dict['prediction_data']
    #scaler_path = dlhub_predictor_dict['scaler_path']
    #training_data_path = dlhub_predictor_dict['training_data_path']
    #servable = DLHubClient().describe_servable(dlhub_servable)

    # Load scaler:
    scaler_path = joblib.load(glob.glob('*preprocessor.pkl')[0])
    # Load model:
    model = joblib.load(glob.glob('*model.pkl')[0])
    # Prediction data comps:
    prediction_data = comp_list
    # Load training data:
    training_data_path = glob.glob('*selected.csv')[0]

    #scaler_path = '/Users/ryanjacobs/'+servable['dlhub']['files']['other'][0]
    #training_data_path = '/Users/ryanjacobs/'+servable['dlhub']['files']['other'][1]
    pred_dict = make_prediction(model, prediction_data, scaler_path, training_data_path, exclude_columns=['composition', 'band_gap'])
    return pred_dict