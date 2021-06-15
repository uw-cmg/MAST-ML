import unittest
import numpy as np
import pandas as pd
import os
import sys

from mastml.data_splitters import SklearnDataSplitter
from mastml.datasets import LocalDatasets
from mastml.models import SklearnModel
from mastml.preprocessing import SklearnPreprocessor
from mastml.baseline_tests import Baseline_tests

sys.path.insert(0, os.path.abspath('../../../'))

from mastml.feature_selectors import NoSelect, EnsembleModelFeatureSelector, PearsonSelector, MASTMLFeatureSelector
from sklearn.ensemble import RandomForestRegressor

class test_baseline(unittest.TestCase):
    def test_baseline_mean(self):
        target = 'E_regression.1'
        extra_columns = ['Material compositions 1', 'Material compositions 2', 'Hop activation barrier', 'E_regression']
        d = LocalDatasets(file_path='mastml/data/figshare_7418492/All_Model_Data.xlsx',
                          target=target,
                          extra_columns=extra_columns,
                          group_column='Material compositions 1',
                          as_frame=True)
        data_dict = d.load_data()
        X = data_dict['X']
        y = data_dict['y']
        # model = SklearnModel(model='RandomForestRegressor', n_estimators=150)
        # preprocessor = SklearnPreprocessor(preprocessor='StandardScaler', as_frame=True)

        # X = pd.DataFrame(np.random.uniform(low=0.0, high=100, size=(50, 5)))
        # y = pd.Series(np.random.uniform(low=0.0, high=100, size=(50,)))
        model = SklearnModel(model='LinearRegression')
        model.fit(X=X, y=y)

        baseline = Baseline_tests()
        baseline.test_mean(X=X, y=y , model=model)
        return

        # test_mean = baseline_tests()
        # test_mean.test_mean(X,y)

if __name__=='__main__':
    unittest.main()