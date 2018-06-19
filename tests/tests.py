import unittest
import random

import pandas as pd
import numpy as np


from mastml import mastml
from mastml import plot_helper, html_helper
from matplotlib.ticker import MaxNLocator



class SmokeTests(unittest.TestCase):

    def test_classification(self):
        mastml.mastml_run('tests/conf/classification.conf', 'tests/csv/three_clusters.csv',
                'results/classification')

    def test_regression(self):
        mastml.mastml_run('tests/conf/regression.conf', 'tests/csv/boston_housing.csv', 'results/regression')

class TestPlots(unittest.TestCase):
    """ don't mind the mismatched naming conventions for [true actual y_true] and [pred prediction
    y_pred] """

    def test_plot_predicted_vs_true(self):
        predicted = np.arange(30) + np.random.random_sample((30,)) * 10 - 3
        actual = np.arange(30)
        stats = {'foo': 500000, 'bar': 123.4566, 'baz': 'impossoble', 'rosco': 123e-500}

        plot_helper.plot_predicted_vs_true(actual, predicted, 'pred-vs-true.png', stats)

    def test_residuals_histogram(self):
        predicted = np.arange(30) + sum(np.random.random_sample((30,)) for _ in range(10)) - 3
        actual = np.arange(30)
        stats = {'foo': 500000, 'bar': 123.4566, 'baz': 'impossoble', 'rosco': 123e-500}

        plot_helper.plot_residuals_histogram(actual, predicted, 'rh.png', stats)

    def test_confusion_matrix(self):
        true = np.random.randint(4, size=(50,))
        pred = true.copy()
        slices = [not bool(x) for x in np.random.randint(3, size=50)]
        pred[slices] = [random.randint(1, 3) for s in slices if s]

        names = ['a', 'b', 'c', 'f']
        pred = np.array([names[x] for x in pred] + ['a', 'a', 'a'])
        true = np.array([names[x] for x in true] + ['b', 'b', 'b'])

        stats = {'foo': 500000, 'bar': 123.4566, 'baz': 'impossoble', 'rosco': 123e-500}

        plot_helper.plot_confusion_matrix(true, pred, 'cf.png', stats)

class TestHtml(unittest.TestCase):

    def test_image_list(self):
        #imgs = ['cf.png', 'rh.png', 'pred-vs-true.png']
        #html_helper.make_html(imgs, 'tests/csv/three_clusters.csv', 'tests/conf/fullrun.conf', 'oop.txt', './')
        html_helper.make_html('results/classification')
        html_helper.make_html('results/regression')

        

class TestRandomizer(unittest.TestCase):

    def test_shuffle_data(self):
        d = pd.DataFrame(np.arange(10).reshape(5,2))
        d.columns = ['a', 'b']
        r = Randomizer('b')
        r.fit(d)
        buffler = r.transform(d)
        good = r.inverse_transform(buffler)
        # this has a 1/120 chance of failing unexpectedly.
        self.assertFalse(d.equals(buffler))
        self.assertTrue(d.equals(good))

class TestNormalization(unittest.TestCase):

    def  test_normalization(self):
        d1 = pd.DataFrame(np.random.random((7,3)), columns=['a','b','c']) * 4 + 6 # some random data

        fn = FeatureNormalization(features=['a','c'], mean=-2, stdev=5)
        fn.fit()

        d2 = fn.transform(d1)
        self.assertTrue(set(d1.columns) == set(d2.columns))
        arr = d2[['a','c']].values
        #import pdb; pdb.set_trace()
        self.assertTrue(abs(arr.mean() - (-2)) < 0.001)
        self.assertTrue(abs(arr.std() - 5) < 0.001)

        d3 = fn.inverse_transform(d2)
        self.assertTrue(set(d1.columns) == set(d3.columns))
        self.assertTrue((abs(d3 - d1) < .001).all().all())
