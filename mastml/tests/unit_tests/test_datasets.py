import unittest
import numpy as np
import pandas as pd
import os
import sys
import shutil
sys.path.insert(0, os.path.abspath('../../../'))

from mastml.datasets import SklearnDatasets, LocalDatasets, FoundryDatasets, MatminerDatasets
import mastml
try:
    mastml_path = mastml.__path__._path[0]
except:
    mastml_path = mastml.__path__[0]

class TestDatasets(unittest.TestCase):

    def test_sklearn(self):
        sklearndata = SklearnDatasets(return_X_y=True, as_frame=True)
        bostonX, bostony = sklearndata.load_boston()
        irisX, irisy = sklearndata.load_iris()
        digitsX, digitsy = sklearndata.load_digits()
        diabetesX, diabetesy = sklearndata.load_diabetes()
        breast_cancerX, breast_cancery = sklearndata.load_breast_cancer()
        wineX, winey = sklearndata.load_wine()
        linnerudX, linnerudy = sklearndata.load_linnerud()
        self.assertEqual(bostonX.shape, (506, 13))
        self.assertEqual(irisX.shape, (150, 4))
        self.assertEqual(digitsX.shape, (1797, 64))
        self.assertEqual(diabetesX.shape, (442, 10))
        self.assertEqual(breast_cancerX.shape, (569, 30))
        self.assertEqual(wineX.shape, (178, 13))
        self.assertEqual(linnerudX.shape, (20, 3))
        return

    '''
    def test_figshare(self):
        FigshareDatasets().download_data(article_id='7418492', savepath=os.getcwd())
        self.assertTrue(os.path.exists('figshare_7418492'))
        return
    '''

    def test_local(self):
        target = 'E_regression.1'
        extra_columns = ['E_regression', 'Material compositions 1', 'Material compositions 2', 'Hop activation barrier']
        file_path = os.path.join(mastml_path, 'data/figshare_7418492/All_Model_Data.xlsx')
        d = LocalDatasets(file_path=file_path,
                          target=target,
                          extra_columns=extra_columns,
                          as_frame=True)
        data_dict = d.load_data()
        X = data_dict['X']
        y = data_dict['y']
        self.assertEqual(X.shape, (408, 287))
        self.assertEqual(y.shape, (408,))
        return

    def test_matminer(self):
        matminerdata = MatminerDatasets()
        df = matminerdata.download_data(name='dielectric_constant', save_data=True)
        self.assertTrue(os.path.exists('dielectric_constant.xlsx'))
        self.assertTrue(os.path.exists('dielectric_constant.pickle'))
        self.assertTrue(df.shape, (1056, 16))
        os.remove('dielectric_constant.xlsx')
        os.remove('dielectric_constant.pickle')
        return

    def test_foundry(self):
        foundrydata = FoundryDatasets(no_local_server=False, anonymous=True, test=True)
        foundrydata.download_data(name='pub_57_wu_highthroughput', download=False)
        return

if __name__ == '__main__':
    unittest.main()
