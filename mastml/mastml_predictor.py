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

def make_prediction(
                    X_test,
                    X_train,
                    y_train,
                    model,
                    preprocessor=None,
                    calibration_file=None,
                    featurizers=None,
                    featurize_on=None,
                    domain=None,
                    composition_column=None,
                    *args,
                    **kwargs,
                    ):
    '''
    Method used to take a saved preprocessor, model and calibration file and output predictions and calibrated uncertainties
    on new test data

    Args:
        X_test: (pd.DataFrame or str), dataframe of featurized test data to be used to make prediction, or string of path
            containing featurized test data in .xlsx or .csv format ready for import with pandas. If passing an already
            featurized dataframe, only the features used to fit the original model should be included, and they should be
            in the same order as the training data used to fit the original model.

        X_train: (pd.DataFrame or str), dataframe of training data used to train original model, or string of path
            containing featurized training data in .xlsx or .csv format ready for import with pandas. Used to extract the
            features used in training, to downselect from newly generated features on test data.

        y_train: (pd.DataFrame or str), dataframe of training target data used to train original model, or string of path
            containing training target data in .xlsx or .csv format ready for import with pandas. Used to return the true
            value of a test data point if that point is present in the training data.

        model: (str), path of saved model in .pkl format (e.g., RandomForestRegressor.pkl)

        preprocessor: (str), path of saved preprocessor in .pkl format (e.g., StandardScaler.pkl)

        calibration_file: path of file containing the recalibration parameters (typically recalibration_parameters_average_test.xlsx)

        featurizers: (list), list of strings denoting paths to saved mastml feature generators, e.g., ["myfolder/ElementalFeatureGenerator.pkl", "myfolder/PolynomialFeatureGenerator.pkl"]

        featurize_on: (list), list of strings of column name in X_test to perform featurization on, needs to be same length and in
            same order as featurizers listed above, e.g., ['Composition', ['feature1', 'feature2'] ]

        domain: (list), list of strings denoting filenames of saved domain.pkl objects, e.g., ['domain_gpr.pkl']

        composition_column: (str), string denoting name of X_test column denoting material compositions. Will be needed if assessing domain with "elemental" method.

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

    # Load in the X_test data if it wasn't provided as a dataframe
    if isinstance(X_test, str):
        if '.xlsx' in X_test:
            X_test = pd.read_excel(X_test, engine='openpyxl')
        elif '.csv' in X_test:
            X_test = pd.read_csv(X_test)
        else:
            raise ValueError('You must provide X_test as .xlsx or .csv file, or loaded pandas DataFrame')

    # Load in X_train data so can get columns to use
    if isinstance(X_train, str):
        if '.xlsx' in X_train:
            X_train = pd.read_excel(X_train, engine='openpyxl')
        elif '.csv' in X_train:
            X_train = pd.read_csv(X_train)
        else:
            raise ValueError('You must provide X_train as .xlsx or .csv file, or loaded pandas DataFrame')
    features_to_keep = X_train.columns.tolist()
    #extra_columns = [col for col in X_test.columns.tolist() if col not in features_to_keep]
    #X_extra = X_test[extra_columns]

    # Load in y_train data so can return true values if that data point is queried as test data
    if isinstance(y_train, str):
        if '.xlsx' in y_train:
            y_train = pd.read_excel(y_train, engine='openpyxl')
        elif '.csv' in y_train:
            y_train = pd.read_csv(y_train)
        else:
            raise ValueError('You must provide y_train as .xlsx or .csv file, or loaded pandas DataFrame')

    # Load featurizers
    df_test = X_test
    if featurizers is not None:
        # Load in the featurizers
        for f, f_on in zip(featurizers, featurize_on):
            gen = joblib.load(f)
            gen.featurize_df = pd.DataFrame(X_test[f_on])
            df_test, _ = gen.evaluate(X=df_test, y=pd.Series(np.zeros(shape=df_test.shape[0])), savepath=None, make_new_dir=False)
        df_test = df_test[features_to_keep]
    else:
        df_test = df_test[features_to_keep]

    # Check if any of the featurized rows are in the training data. If so, append the true target value
    # Commented by Lane because of bug
    '''
    y_true_list = list()
    for i, vals_i in enumerate(df_test[features_to_keep].iterrows()):
        found = False
        for j, vals_j in enumerate(X_train[features_to_keep].iterrows()):
            if vals_i[1].round(6).equals(vals_j[1].round(6)):
                y_true_list.append(np.array(y_train)[j][0])
                found = True
                break
        if found == False:
            y_true_list.append(np.nan)
    '''

    # Load preprocessor
    if preprocessor is not None:
        preprocessor = joblib.load(preprocessor)
        df_test = preprocessor.transform(df_test)

    # Check the model is an ensemble and get an error bar:
    ensemble_models = ['RandomForestRegressor',
                       'GradientBoostingRegressor',
                       'BaggingRegressor',
                       'ExtraTreesRegressor',
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

    for col in X_test.columns.tolist():
        if col not in features_to_keep:
            pred_df[col] = X_test[col]

    # Add the y_true column into the predicted dataframe:
    # Commented by Lane because of bug
    #pred_df['y_true'] = y_true_list

    # Concatenate the extra columns to the prediction dataframe
    #pred_df = pd.concat([pred_df, X_extra], axis=1)

    # Evaluate the domain predictions on the test data
    domains_list = list()
    if domain is not None:
        for domain_type in domain:
            domain_check = joblib.load(domain_type)
            if domain_check.check_type == 'elemental':
                if composition_column is None:
                    print("Error: trying to assess domain with 'elemental' method but no composition_column has been specified")
                domains_list.append(domain_check.predict(X_test[composition_column]))
            elif domain_check.check_type == 'madml':
                domains_list.append(domain_check.predict(X_test, *args, **kwargs))
            else:
                domains_list.append(domain_check.predict(df_test))
        domain_df = pd.concat(domains_list, axis=1)
        pred_df = pd.concat([pred_df, domain_df], axis=1)

    return pred_df


def make_prediction_dlhub(input_dict):
    '''
    Method used to take a saved preprocessor, model and calibration file and output predictions and calibrated uncertainties
    on new test data

    Args:
        input_dict: (dict), dictionary of input passed to predictor. The dictionary may have the following keys:

            X_test: (pd.DataFrame or str), dataframe of featurized test data to be used to make prediction, or string of path
                containing featurized test data in .xlsx or .csv format ready for import with pandas. If passing an already
                featurized dataframe, only the features used to fit the original model should be included, and they should be
                in the same order as the training data used to fit the original model.

            featurizers: (list), list of strings denoting paths to saved mastml feature generators, e.g., ["myfolder/ElementalFeatureGenerator.pkl", "myfolder/PolynomialFeatureGenerator.pkl"]

            featurize_on: (list), list of strings of column name in X_test to perform featurization on, needs to be same length and in
                same order as featurizers listed above, e.g., ['Composition', ['feature1', 'feature2'] ]

            composition_column: (str), string denoting name of X_test column denoting material compositions. Will be needed if assessing domain with "elemental" method.

    Returns:
        pred_df: (pd.DataFrame), dataframe containing column of model predictions (y_pred) and, if applicable, calibrated uncertainties (y_err).
            Will also include any extra columns denoted in extra_columns parameter.
    '''

    # Load model:
    model = joblib.load('model.pkl')

    # Check if recalibration params exist:
    if os.path.exists('calibration_file.xlsx'):
         recal_params = pd.read_excel(os.path.join(os.getcwd(), 'calibration_file.xlsx'), engine='openpyxl')
    elif os.path.exists('calibration_file.csv'):
        recal_params = pd.read_csv(os.path.join(os.getcwd(), 'calibration_file.csv'))
    else:
        recal_params = None

    # Load in the X_test data
    X_test = input_dict['X_test']

    # Load in the X_train data
    if os.path.exists('X_train.xlsx'):
        X_train = pd.read_excel('X_train.xlsx', engine='openpyxl')
    elif os.path.exists('X_train.csv'):
        X_train = pd.read_csv('X_train.csv')

    features_to_keep = X_train.columns.tolist()

    # Load in the y_train data
    if os.path.exists('y_train.xlsx'):
        y_train = pd.read_excel('y_train.xlsx', engine='openpyxl')
    elif os.path.exists('y_train.csv'):
        y_train = pd.read_csv('y_train.csv')

    # Load featurizers
    try:
        featurizers = input_dict['featurizers']
        featurize_on = input_dict['featurize_on']
    except:
        featurizers = None
        featurize_on = None
    df_test = X_test
    if featurizers is not None:
        # Load in the featurizers
        for f, f_on in zip(featurizers, featurize_on):
            try:
                gen = joblib.load(f+'.pkl')
                #print('generator', gen)
            except:
                gen = joblib.load(f)
            gen.featurize_df = pd.DataFrame(X_test[f_on])
            df_test, _ = gen.evaluate(X=df_test, y=pd.Series(np.zeros(shape=df_test.shape[0])), savepath=None, make_new_dir=False)
        df_test = df_test[features_to_keep]
    else:
        df_test = df_test[features_to_keep]

    # Check if any of the featurized rows are in the training data. If so, append the true target value
    y_true_list = list()
    for i, vals_i in enumerate(df_test[features_to_keep].iterrows()):
        found = False
        for j, vals_j in enumerate(X_train[features_to_keep].iterrows()):
            if vals_i[1].round(6).equals(vals_j[1].round(6)):
                y_true_list.append(np.array(y_train)[j][0])
                found = True
                break
        if found == False:
            y_true_list.append(np.nan)

    # Load preprocessor
    if os.path.exists('preprocessor.pkl'):
        preprocessor = joblib.load('preprocessor.pkl')
        df_test = preprocessor.transform(df_test)

    # Check the model is an ensemble and get an error bar:
    ensemble_models = ['RandomForestRegressor',
                       'GradientBoostingRegressor',
                       'BaggingRegressor',
                       'ExtraTreesRegressor',
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

    # Add the y_true column into the predicted dataframe:
    pred_df['y_true'] = y_true_list

    # Evaluate the domain predictions on the test data, if such files exist
    files = os.listdir(os.getcwd())
    domain = list()
    for f in files:
        if 'domain_' in f:
            domain.append(f)
    try:
        composition_column = input_dict['composition_column']
    except:
        composition_column = None
    domains_list = list()
    if len(domain) > 0:
        for domain_type in domain:
            domain_check = joblib.load(domain_type)
            if domain_check.check_type == 'elemental':
                if composition_column is None:
                    print("Error: trying to assess domain with 'elemental' method but no composition_column has been specified")
                domains_list.append(domain_check.predict(X_test[composition_column]))
            elif domain_check.check_type == 'madml':
                domains_list.append(domain_check.predict(X_test))
            else:
                domains_list.append(domain_check.predict(df_test))
        domain_df = pd.concat(domains_list, axis=1)
        pred_df = pd.concat([pred_df, domain_df], axis=1)

    for col in X_test.columns.tolist():
        pred_df[col] = X_test[col]

    return pred_df

def make_prediction_dlhub_OLD(input_dict):
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
