from mastml.data_splitters import SklearnDataSplitter, NoSplit
from mastml.preprocessing import SklearnPreprocessor
from mastml.datasets import LocalDatasets
from mastml.models import SklearnModel
from mastml.domain import Domain

from pathlib import Path

import subprocess
import glob
import os

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
X_extra = data_dict['X_extra']
groups = data_dict['groups']
X_testdata = data_dict['X_testdata']

metrics = [
           'r2_score',
           'mean_absolute_error',
           'root_mean_squared_error',
           'rmse_over_stdev',
           ]

preprocessor = SklearnPreprocessor(
                                   preprocessor='StandardScaler',
                                   as_frame=True,
                                   )

model = SklearnModel(model='RandomForestRegressor')

# UQ
splitter = SklearnDataSplitter(
                               splitter='RepeatedKFold',
                               n_repeats=1,
                               n_splits=5
                               )

splitter.evaluate(
                  X=X,
                  y=y,
                  models=[model],
                  preprocessor=preprocessor,
                  metrics=metrics,
                  plots=['Scatter', 'Histogram', 'Error'],
                  X_extra=X_extra,
                  error_method='stdev_weak_learners',
                  recalibrate_errors=True,
                  verbosity=3,
                  )


file_to_move = glob.glob('Ran*')[0]
cal_name = 'calibration_run'
subprocess.run(['mv', file_to_move, cal_name])

# Domain with MADML
params = {'n_repeats': 2}
domain = ('madml', params)

splitter = NoSplit()
splitter.evaluate(
                  X=X,
                  y=y,
                  models=[model],
                  preprocessor=preprocessor,
                  metrics=metrics,
                  plots=['Scatter', 'Histogram'],
                  X_extra=X_extra,
                  verbosity=3,
                  domain=[domain],
                  )

file_to_move = glob.glob('Ran*')[0]
dom_name = 'domain_run'
subprocess.run(['mv', file_to_move, dom_name])

cal_params = os.path.join(cal_name, 'recalibration_parameters_train.csv')
model_path = os.path.join(dom_name, 'RandomForestRegressor.pkl')
preprocessor_path = os.path.join(dom_name, 'StandardScaler.pkl')
domain_path = list(map(str, Path(dom_name).rglob('domain_*.pkl')))

# Create a container
