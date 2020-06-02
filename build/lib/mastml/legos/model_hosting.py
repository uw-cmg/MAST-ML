import pandas as pd
import numpy as np
from dlhub_sdk import DLHubClient
from dlhub_sdk.models.servables.sklearn import ScikitLearnModel
from dlhub_sdk.models.servables.python import PythonStaticMethodModel
from sklearn.externals import joblib
from mastml.legos.dlhub_predictor import run_dlhub_prediction
import os
import mastml
import logging
import shutil

log = logging.getLogger('mastml')

def get_input_columns(training_data_path, exclude_columns):
    # Load in training data and get input columns
    try:
        df_train = pd.read_csv(training_data_path)
    except:
        df_train = pd.read_excel(training_data_path)
    input_columns = [col for col in df_train.columns.tolist() if col not in exclude_columns]
    return input_columns

def host_model(model_path, preprocessor_path, training_data_path,
               model_title, model_name, model_type="scikit-learn"):
    # Assume the model will be hosted with a list of the input column names
    # input_columns
    # Also assume the model will be hosted with the needed preprocessing routine
    # scaler_path

    #input_columns = get_input_columns(training_data_path=training_data_path, exclude_columns=exclude_columns)
    #n_input_columns = len(input_columns)

    dl = DLHubClient()
    # Create the model from saved .pkl file. This one was from mastml run
    if model_type == 'scikit-learn':
        #model_info = ScikitLearnModel.create_model(model_path,
        #                                       n_input_columns=n_input_columns,
        #                                       serialization_method=serialization_method)
        model = PythonStaticMethodModel.from_function_pointer(run_dlhub_prediction)
    else:
        raise ValueError("Only scikit-learn models supported at this time")

    # Some model descriptive info
    model.set_name(model_name).set_title(model_title)
    #model_info.set_domains(["materials science"])

    # Describe the inputs/outputs
    model.set_inputs('list', 'list of material compositions to predict', item_type='string')
    model.set_outputs(data_type='float', description='Predicted value from trained sklearn model')

    # Add additional files to model servable- needed to do featurization of predictions using DLHub
    log.info('Submitting preprocessor file to DLHub:')
    log.info(os.path.abspath(preprocessor_path))
    log.info('Submitting model file to DLHub:')
    log.info(os.path.abspath(model_path))
    log.info('Submitting training data file to DLHub:')
    log.info(os.path.abspath(training_data_path))
    log.info('Submitting mastml directory to DLHub:')
    log.info(os.path.join(os.path.abspath(mastml.__path__[0])))
    # Need to change model, preprocessor names to be standard model.pkl and preprocessor.pkl names. Copy them and change names
    model_dirname = os.path.dirname(model_path)
    #shutil.copy(model_path, os.path.join(model_dirname, 'model.pkl'))
    shutil.copy(model_path, os.path.join(os.getcwd(), 'model.pkl'))
    preprocessor_dirname = os.path.dirname(preprocessor_path)
    #shutil.copy(preprocessor_path, os.path.join(preprocessor_dirname, 'preprocessor.pkl'))
    shutil.copy(preprocessor_path, os.path.join(os.getcwd(), 'preprocessor.pkl'))
    #model_path = os.path.join(model_dirname, 'model.pkl')
    #preprocessor_path = os.path.join(preprocessor_dirname, 'preprocessor.pkl')
    shutil.copy(training_data_path, os.path.join(os.getcwd(), 'selected.csv'))
    model.add_directory(os.path.join(os.path.abspath(mastml.__path__[0])), recursive=True)
    #model.add_file(os.path.abspath(model_path))
    #model.add_file(os.path.abspath(preprocessor_path))  # Add the preprocessor .pkl file
    #model.add_file(os.path.abspath(training_data_path)) # Add the training_data .csv file
    model.add_file('model.pkl')
    model.add_file('preprocessor.pkl')
    model.add_file('selected.csv')

    # Add pip installable dependency for MAST-ML
    model.add_requirement('mastml', 'latest')

    res = dl.publish_servable(model)
    return dl, res



