import unittest
import os
from mastml.datasets import LocalDatasets
from mastml.models import SklearnModel
from mastml.preprocessing import SklearnPreprocessor
from mastml.baseline_tests import Baseline_tests
from mastml.datasets import SklearnDatasets
from sklearn.model_selection import train_test_split

import mastml
try:
    mastml_path = mastml.__path__._path[0]
except:
    mastml_path = mastml.__path__[0]

class test_baseline(unittest.TestCase):

    def test_baseline_mean(self):
        target = 'E_regression.1'
        extra_columns = ['Material compositions 1', 'Material compositions 2', 'Hop activation barrier', 'E_regression']
        d = LocalDatasets(file_path=os.path.join(mastml_path, 'data/figshare_7418492/All_Model_Data.xlsx'),
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
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        model.fit(X=X_train, y=y_train)

        baseline = Baseline_tests()
        baseline.test_mean(X_train, X_test, y_train, y_test, model)
        os.remove("data_preprocessed_.csv")
        os.remove("StandardScaler.pkl")
        return

    def test_baseline_permute(self):
        target = 'E_regression.1'
        extra_columns = ['Material compositions 1', 'Material compositions 2', 'Hop activation barrier', 'E_regression']
        d = LocalDatasets(file_path=os.path.join(mastml_path, 'data/figshare_7418492/All_Model_Data.xlsx'),
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
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        model.fit(X = X_train, y = y_train)

        baseline = Baseline_tests()
        baseline.test_permuted(X_train, X_test, y_train, y_test, model)
        os.remove("data_preprocessed_.csv")
        os.remove("StandardScaler.pkl")
        return

    def test_baseline_nearest_neighbor_kdTree(self):
        target = 'E_regression.1'
        extra_columns = ['Material compositions 1', 'Material compositions 2', 'Hop activation barrier', 'E_regression']
        d = LocalDatasets(file_path=os.path.join(mastml_path, 'data/figshare_7418492/All_Model_Data.xlsx'),
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
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        model.fit(X = X_train, y = y_train)

        baseline = Baseline_tests()
        baseline.test_nearest_neighbour_kdtree(X_train, X_test, y_train.tolist(), y_test, model)
        os.remove("data_preprocessed_.csv")
        os.remove("StandardScaler.pkl")
        return

    def test_baseline_nearest_neighbor_cdist(self):
        target = 'E_regression.1'
        extra_columns = ['Material compositions 1', 'Material compositions 2', 'Hop activation barrier', 'E_regression']
        d = LocalDatasets(file_path=os.path.join(mastml_path, 'data/figshare_7418492/All_Model_Data.xlsx'),
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
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        model.fit(X = X_train, y = y_train)

        baseline = Baseline_tests()
        baseline.test_nearest_neighbour_cdist(X_train, X_test, y_train.tolist(), y_test, model)
        os.remove("data_preprocessed_.csv")
        os.remove("StandardScaler.pkl")
        return

    def test_baseline_classifier_random(self):
        X, y = SklearnDatasets(as_frame=True).load_iris()
        model = SklearnModel(model="KNeighborsClassifier")
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        model.fit(X_train,y_train)
        baseline = Baseline_tests()
        baseline.test_classifier_random(X_train, X_test, y_train, y_test, model)
        return

    def test_baseline_classifier_dominant(self):
        X, y = SklearnDatasets(as_frame=True).load_iris()
        model = SklearnModel(model="KNeighborsClassifier")
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        model.fit(X_train,y_train)

        baseline = Baseline_tests()
        baseline.test_classifier_random(X_train, X_test, y_train, y_test, model)
        return

if __name__=='__main__':
    unittest.main()