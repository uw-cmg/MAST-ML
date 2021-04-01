import unittest
import numpy as np
import pandas as pd
import os
import sys
sys.path.insert(0, os.path.abspath('../../../'))

from mastml.plots import Scatter, Histogram

class TestPlots(unittest.TestCase):

    def test_scatter(self):
        X = pd.Series(np.random.uniform(low=0.0, high=100, size=(50,)))
        y = pd.Series(np.random.uniform(low=0.0, high=100, size=(50,)))

        Scatter().plot_predicted_vs_true(y_true=X,
                                         y_pred=y,
                                         savepath=os.getcwd(),
                                         x_label='TEST_scatter',
                                         data_type='test',)
        self.assertTrue(os.path.exists('parity_plot_test.png'))
        os.remove('parity_plot_test.png')
        return

    def test_histogram(self):
        X = pd.Series(np.random.uniform(low=0.0, high=100, size=(50,)))

        Histogram().plot_histogram(df=X,
                                   savepath=os.getcwd(),
                                   file_name='TEST_hist',
                                   x_label='TEST_hist')
        self.assertTrue(os.path.exists('TEST_hist.png'))
        self.assertTrue(os.path.exists('TEST_hist.xlsx'))
        self.assertTrue(os.path.exists('TEST_hist_statistics.xlsx'))
        os.remove('TEST_hist.png')
        os.remove('TEST_hist.xlsx')
        os.remove('TEST_hist_statistics.xlsx')
        return

    def test_residual_histogram(self):
        X = pd.Series(np.random.uniform(low=0.0, high=100, size=(50,)))
        y = pd.Series(np.random.uniform(low=0.0, high=100, size=(50,)))

        Histogram().plot_residuals_histogram(y_true=X,
                                             y_pred=y,
                                             savepath=os.getcwd())
        self.assertTrue(os.path.exists('residual_histogram.png'))
        self.assertTrue(os.path.exists('residual_histogram.xlsx'))
        self.assertTrue(os.path.exists('residual_histogram_statistics.xlsx'))
        os.remove('residual_histogram.png')
        os.remove('residual_histogram.xlsx')
        os.remove('residual_histogram_statistics.xlsx')
        return


if __name__ == '__main__':
    unittest.main()
