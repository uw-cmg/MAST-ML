from pathlib import Path

import subprocess
import shutil
import glob
import os

cal_name = 'calibration_run'
dom_name = 'domain_run'

cal_params = os.path.join(cal_name, 'recalibration_parameters_train.csv')
model_path = os.path.join(dom_name, 'RandomForestRegressor.pkl')
preprocessor_path = os.path.join(dom_name, 'StandardScaler.pkl')
domain_path = list(map(str, Path(dom_name).rglob('domain_*.pkl')))

files = [cal_params, model_path, preprocessor_path, *domain_path]

output = 'container_files'
os.makedirs(output, exist_ok=True)

for f in files:
    shutil.copy(f, os.path.join(output, os.path.basename(f)))
