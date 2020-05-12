import pandas as pd
import numpy as np
from dlhub_sdk import DLHubClient
from dlhub_sdk.models.servables.sklearn import ScikitLearnModel
from sklearn.externals import joblib
import os
import mastml
import logging

log = logging.getLogger('mastml')

def get_input_columns(training_data_path, exclude_columns):
    # Load in training data and get input columns
    try:
        df_train = pd.read_csv(training_data_path)
    except:
        df_train = pd.read_excel(training_data_path)
    input_columns = [col for col in df_train.columns.tolist() if col not in exclude_columns]
    return input_columns

def host_model(model_path, preprocessor_path, training_data_path, exclude_columns,
               set_title, model_name, serialization_method="joblib", model_type="scikit-learn"):
    # Assume the model will be hosted with a list of the input column names
    # input_columns
    # Also assume the model will be hosted with the needed preprocessing routine
    # scaler_path

    input_columns = get_input_columns(training_data_path=training_data_path, exclude_columns=exclude_columns)
    n_input_columns = len(input_columns)

    dl = DLHubClient()
    # Create the model from saved .pkl file. This one was from mastml run
    if model_type == 'scikit-learn':
        model_info = ScikitLearnModel.create_model(model_path,
                                               n_input_columns=n_input_columns,
                                               serialization_method=serialization_method)
    else:
        raise ValueError("Only scikit-learn models supported at this time")

    # Some model descriptive info
    model_info.set_title(set_title)
    model_info.set_name(model_name)
    #model_info.set_domains(["materials science"])

    # Add additional files to model servable- needed to do featurization of predictions using DLHub
    model_info.add_file(os.path.abspath(preprocessor_path))  # Add the preprocessor .pkl file
    model_info.add_file(os.path.abspath(training_data_path)) # Add the training_data .csv file
    model_info.add_file(os.path.abspath(os.path.join(mastml.__path__[0],'legos/dlhub_predictor.py')))  # Add the dlhub_predictor.py script, which featurizes prediction data and runs the DLHub servable to make predictions

    log.info('Submitting preprocessor file to DLHub:')
    log.info(os.path.abspath(preprocessor_path))
    log.info('Submitting training data file to DLHub:')
    log.info(os.path.abspath(training_data_path))

    # Add pip installable dependency for MAST-ML
    model_info.add_requirement('mastml', 'latest')

    res = dl.publish_servable(model_info)
    return dl, res



