"""
This module contains methods for easily making new predictions on test data once a suitable model has been trained. Also
available is output of calibrated uncertainties for each prediction.

make_prediction:
    Method used to take a saved preprocessor, model and calibration file and output predictions and calibrated uncertainties
    on new test data.
"""

import pandas as pd
import joblib
import numpy as np
import os
from mastml import feature_generators

def make_prediction(X_test, model, X_test_extra=None, preprocessor=None, calibration_file=None, featurize=False,
                    featurizer=None, features_to_keep=None, featurize_on=None, **kwargs):
    '''
    Method used to take a saved preprocessor, model and calibration file and output predictions and calibrated uncertainties
    on new test data

    Args:
        X_test: (pd.DataFrame or str), dataframe of featurized test data to be used to make prediction, or string of path
            containing featurized test data in .xlsx or .csv format ready for import with pandas. Only the features used
            to fit the original model should be included, and they should be in the same order as the training data used
            to fit the original model.

        model: (str), path of saved model in .pkl format (e.g., RandomForestRegressor.pkl)

        X_test_extra: (pd.DataFrame, list or str), dataframe containing the extra data associated with X_test, or a
            list of strings denoting extra columns present in X_test not to be used in prediction.
            If a string is provided, it is interpreted as a path to a .xlsx or .csv file containing the extra column data

        preprocessor: (str), path of saved preprocessor in .pkl format (e.g., StandardScaler.pkl)

        calibration_file: path of file containing the recalibration parameters (typically recalibration_parameters_average_test.xlsx)

        featurize: (bool), whether or not featurization of the provided X_test data needs to be performed

        featurizer: (str), string denoting a mastml.feature_generators class, e.g., "ElementalFeatureGenerator"

        features_to_keep: (list), list of strings denoting column names of features to keep for running model prediction

        featurize_on: (str), string of column name in X_test to perform featurization on

        **kwargs: additional key-value pairs of parameters for feature generator, e.g., composition_df=composition_df['Compositions'] if
            running ElementalFeatureGenerator

    Returns:
        pred_df: (pd.DataFrame), dataframe containing column of model predictions (y_pred) and, if applicable, calibrated uncertainties (y_err).
            Will also include any extra columns denoted in extra_columns parameter.
    '''

    # Load model:
    model = joblib.load(model)

    # Check if recalibration params exist:
    if calibration_file is not None:
        if '.xlsx' in calibration_file:
            recal_params = pd.read_excel(calibration_file, engine='openpyxl')
        elif '.csv' in calibration_file:
            recal_params = pd.read_csv(calibration_file)
        else:
            raise ValueError('calibration_file should be either a .csv or .xlsx file to be loaded using pandas')
    else:
         recal_params = None

    if isinstance(X_test, str):
        if '.xlsx' in X_test:
            X_test = pd.read_excel(X_test, engine='openpyxl')
        elif '.csv' in X_test:
            X_test = pd.read_csv(X_test)
        else:
            raise ValueError('You must provide X_test as .xlsx or .csv file, or loaded pandas DataFrame')

    if X_test_extra is not None:
        if isinstance(X_test_extra, str):
            if '.xlsx' in X_test_extra:
                df_extra = pd.read_excel(X_test_extra, engine='openpyxl')
            elif '.csv' in X_test_extra:
                df_extra = pd.read_csv(X_test_extra)
        elif isinstance(X_test_extra, list):
            df_extra = X_test[X_test_extra]
            X_test = X_test.drop(X_test_extra, axis=1)
        else:
            # Assume a dataframe was passed in
            df_extra = X_test_extra

    if featurize == False:
        df_test = X_test
    else:
        featurizer = getattr(feature_generators, featurizer)(**kwargs)
        df_test, _ = featurizer.fit_transform(X_test[featurize_on])
        df_test = df_test[features_to_keep]

    # Load preprocessor
    if preprocessor is not None:
        preprocessor = joblib.load(preprocessor)
        df_test = preprocessor.transform(df_test)

    # Check the model is an ensemble and get an error bar:
    ensemble_models = ['RandomForestRegressor','GradientBoostingRegressor','BaggingRegressor','ExtraTreesRegressor','AdaBoostRegressor']
    try:
        model_name = model.model.__class__.__name__
    except:
        model_name = model.__class__
    yerr = list()    
    if model_name in ensemble_models:
        X_aslist = df_test.values.tolist()
        for x in range(len(X_aslist)):
            preds = list()
            if model_name == 'RandomForestRegressor':
                for pred in model.model.estimators_:
                    preds.append(pred.predict(np.array(X_aslist[x]).reshape(1, -1))[0])
            elif model_name == 'BaggingRegressor':
                for pred in model.model.estimators_:
                    preds.append(pred.predict(np.array(X_aslist[x]).reshape(1, -1))[0])
            elif model_name == 'ExtraTreesRegressor':
                for pred in model.model.estimators_:
                    preds.append(pred.predict(np.array(X_aslist[x]).reshape(1, -1))[0])
            elif model_name == 'GradientBoostingRegressor':
                for pred in model.model.estimators_.tolist():
                    preds.append(pred[0].predict(np.array(X_aslist[x]).reshape(1, -1))[0])
            elif model_name == 'AdaBoostRegressor':
                for pred in model.model.estimators_:
                    preds.append(pred.predict(np.array(X_aslist[x]).reshape(1, -1))[0])
            if recal_params is not None:
                yerr.append(recal_params['a'][0]*np.std(preds)+recal_params['b'][0])
            else:            
                yerr.append(np.std(preds))

    if model_name == 'GaussianProcessRegressor':
        y_pred_new, yerr = model.model.predict(df_test, return_std=True)
    else:
        y_pred_new = model.predict(df_test)

    if len(yerr) > 0:
        pred_df = pd.DataFrame(y_pred_new, columns=['y_pred'])
        pred_df['y_err'] = yerr
    else:
        pred_df = pd.DataFrame(y_pred_new, columns=['y_pred'])

    if X_test_extra is not None:
        for col in df_extra.columns.tolist():
            pred_df[col] = df_extra[col]

    return pred_df


def make_prediction_dlhub(input_dict):
    '''
    Prediction script, same functionality as make_prediction above, but tailored for model running on DLHub/Foundry

    Use this function as the function pointer for DLHub PythonStaticMethodModel (see 'Foundry_model_upload_example.ipynb'
    in main mastml folder)

    Things that need to be uploaded:
        model.pkl (must have this name)
        X_train.xlsx (or X_train.csv) (must have this name)
    Optional to upload:
        preprocessor.pkl (must have this name)

    Args:
        input_dict (dict): dictionary containing at least the following: {'X_test': pd.DataFrame() of featurized test data}
                            The keys are the input arguments of mastml_predictor (see above for explanation)
    '''
    X_test = input_dict['X_test']
    if 'X_test_extra' in input_dict.keys():
        X_test_extra = input_dict['X_test_extra']
    else:
        X_test_extra = None
    if 'featurize' in input_dict.keys():
        featurize = input_dict['featurize']
    else:
        featurize = False
    if 'featurizer' in input_dict.keys():
        featurizer = input_dict['featurizer']
    if 'featurize_on' in input_dict.keys():
        featurize_on = input_dict['featurize_on']

    kwargs = dict()
    main_keys = ['X_test', 'X_test_extra', 'featurize', 'featurizer', 'featurize_on']
    for k, v in input_dict.items():
        if k not in main_keys:
            kwargs[k] = v

    # Load model:
    model = joblib.load(os.path.join(os.getcwd(), 'model.pkl'))

    # Load training data:
    if os.path.exists('X_train.xlsx'):
        X_train = pd.read_excel(os.path.join(os.getcwd(), 'X_train.xlsx'), engine='openpyxl')
    elif os.path.exists('X_train.csv'):
        X_train = pd.read_csv(os.path.join(os.getcwd(), 'X_train.csv'))
    features_to_keep = X_train.columns.tolist()

    # Check if recalibration params exist:
    if os.path.exists('calibration_file.xlsx'):
         recal_params = pd.read_excel(os.path.join(os.getcwd(), 'calibration_file.xlsx'), engine='openpyxl')
    elif os.path.exists('calibration_file.csv'):
        recal_params = pd.read_csv(os.path.join(os.getcwd(), 'calibration_file.csv'))
    else:
        recal_params = None

    if featurize == False:
        df_test = X_test
    else:
        featurizer = getattr(feature_generators, featurizer)(**kwargs)
        df_test, _ = featurizer.fit_transform(X_test[featurize_on])
        df_test = df_test[features_to_keep]

    # Load preprocessor
    if os.path.exists('preprocessor.pkl'):
        preprocessor = joblib.load(os.path.join(os.getcwd(), 'preprocessor.pkl'))
        df_test = preprocessor.transform(df_test)

    # Check the model is an ensemble and get an error bar:
    ensemble_models = ['RandomForestRegressor', 'GradientBoostingRegressor', 'BaggingRegressor', 'ExtraTreesRegressor',
                       'AdaBoostRegressor']
    try:
        model_name = model.model.__class__.__name__
    except:
        model_name = model.__class__
    yerr = list()
    if model_name in ensemble_models:
        X_aslist = df_test.values.tolist()
        for x in range(len(X_aslist)):
            preds = list()
            if model_name == 'RandomForestRegressor':
                for pred in model.model.estimators_:
                    preds.append(pred.predict(np.array(X_aslist[x]).reshape(1, -1))[0])
            elif model_name == 'BaggingRegressor':
                for pred in model.model.estimators_:
                    preds.append(pred.predict(np.array(X_aslist[x]).reshape(1, -1))[0])
            elif model_name == 'ExtraTreesRegressor':
                for pred in model.model.estimators_:
                    preds.append(pred.predict(np.array(X_aslist[x]).reshape(1, -1))[0])
            elif model_name == 'GradientBoostingRegressor':
                for pred in model.model.estimators_.tolist():
                    preds.append(pred[0].predict(np.array(X_aslist[x]).reshape(1, -1))[0])
            elif model_name == 'AdaBoostRegressor':
                for pred in model.model.estimators_:
                    preds.append(pred.predict(np.array(X_aslist[x]).reshape(1, -1))[0])
            if recal_params is not None:
                yerr.append(recal_params['a'][0] * np.std(preds) + recal_params['b'][0])
            else:
                yerr.append(np.std(preds))

    if model_name == 'GaussianProcessRegressor':
        y_pred_new, yerr = model.model.predict(df_test, return_std=True)
    else:
        y_pred_new = model.predict(df_test)

    if len(yerr) > 0:
        pred_df = pd.DataFrame(y_pred_new, columns=['y_pred'])
        pred_df['y_err'] = yerr
    else:
        pred_df = pd.DataFrame(y_pred_new, columns=['y_pred'])

    if X_test_extra is not None:
        for col in X_test_extra.columns.tolist():
            pred_df[col] = X_test_extra[col]

    return pred_df

