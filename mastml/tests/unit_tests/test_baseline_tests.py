import unittest
import os
import sys
from mastml.datasets import LocalDatasets
from mastml.models import SklearnModel
from mastml.preprocessing import SklearnPreprocessor
from mastml.baseline_tests import Baseline_tests

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
        preprocessor = SklearnPreprocessor(preprocessor='StandardScaler', as_frame=True)
        model = SklearnModel(model='LinearRegression')
        X = preprocessor.evaluate(X,y)
        model.fit(X=X, y=y)

        baseline = Baseline_tests()
        baseline.test_mean(X=X, y=y , model=model)
        return

    def test_baseline_permute(self):
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
        preprocessor = SklearnPreprocessor(preprocessor='StandardScaler', as_frame=True)
        model = SklearnModel(model='LinearRegression')
        X = preprocessor.evaluate(X,y)
        model.fit(X=X, y=y)

        baseline = Baseline_tests()
        baseline.test_permuted(X=X, y=y , model=model)
        return

    def test_baseline_nearest_neighbor_kdTree(self):
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
        preprocessor = SklearnPreprocessor(preprocessor='StandardScaler', as_frame=True)
        model = SklearnModel(model='LinearRegression')
        X = preprocessor.evaluate(X,y)
        model.fit(X=X, y=y)

        baseline = Baseline_tests()
        baseline.test_nearest_neighbour_kdtree(X=X, y=y, model=model)
        return

    def test_baseline_nearest_neighbor_cdist(self):
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
        preprocessor = SklearnPreprocessor(preprocessor='StandardScaler', as_frame=True)
        model = SklearnModel(model='LinearRegression')
        X = preprocessor.evaluate(X,y)
        model.fit(X=X, y=y)

        baseline = Baseline_tests()
        baseline.test_nearest_neighbour_cdist(X=X, y=y, model=model, d_metric="euclidean")
        return


if __name__=='__main__':
    unittest.main()