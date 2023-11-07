from mastml.mastml_predictor import make_prediction
from pathlib import Path
import pandas as pd


def main():

    # Read important files
    X_test = pd.read_csv('/mnt/test.csv')  # The test features
    X_train = pd.read_csv('X_train.csv')  # The training features
    y_train = pd.read_csv('y_train.csv')  # The training target
    cal_params = 'recalibration_parameters_train.csv'  # Calibration parameters
    model_path = 'RandomForestRegressor.pkl'  # The trained regression model
    preprocessor_path = 'StandardScaler.pkl'  # The trained scaler model
    domain_path = list(map(str, Path('./').rglob('domain_*.pkl')))  # Domain methods

    # Use custom thresholds for madml domain (delete for default)
    thresholds = [('residual', 0.75), ('uncertainty', 0.2)]

    pred_df = make_prediction(
                              X_train=X_train,
                              y_train=y_train,
                              X_test=X_test,
                              model=model_path,
                              preprocessor=preprocessor_path,
                              domain=domain_path,
                              madml_thresholds=thresholds
                              )

    # Save the predictions into the mounted directory
    pred_df.to_csv('/mnt/prediction.csv', index=False)

    return pred_df


if __name__ == '__main__':
    main()
