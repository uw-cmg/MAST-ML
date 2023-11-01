from mastml.mastml_predictor import make_prediction
from mastml.datasets import LocalDatasets
from pathlib import Path
import os

path_fullfit = './output/split_0'
target = 'E_regression_shift'
extra_columns = ['mat', 'group']

d = LocalDatasets(
                  file_path='./diffusion.csv',
                  target=target,
                  extra_columns=extra_columns,
                  testdata_columns=None,
                  as_frame=True
                  )

# Load the data with the load_data() method
data_dict = d.load_data()

# Let's assign each data object to its respective name
X = data_dict['X']
y = data_dict['y']

model_path = os.path.join(path_fullfit, 'RandomForestRegressor.pkl')
preprocessor_path = os.path.join(path_fullfit, 'StandardScaler.pkl')
domain_path = list(map(str, Path(path_fullfit).rglob('domain_*.pkl')))

pred_df = make_prediction(
                          X_train=X,
                          y_train=y,
                          X_test=X,
                          model=model_path,
                          preprocessor=preprocessor_path,
                          domain=domain_path,
                          )

print(pred_df)
