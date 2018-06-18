import unittest
import pandas as pd
import numpy as np


from mastml import mastml


class SmokeTests(unittest.TestCase):

    def test_classification(self):
        mastml.mastml_run('tests/conf/fullrun.conf', 'tests/csv/three_clusters.csv',
                'results/classification')

    def test_regression(self):
        mastml.mastml_run('tests/conf/regression.conf', 'tests/csv/boston_housing.csv', 'results/regression')


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
