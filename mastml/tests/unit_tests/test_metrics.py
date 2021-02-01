import unittest
import numpy as np
import pandas as pd
import os
import sys
sys.path.insert(0, os.path.abspath('../../../'))

from mastml.metrics import Metrics

class TestMetrics(unittest.TestCase):

    def test_metrics(self):
        y = pd.Series(np.random.uniform(low=0.0, high=100, size=(50,)))
        metrics = Metrics(metrics_list=list())
        all_metrics = metrics._metric_zoo()
        metrics_list = all_metrics.keys()
        stats_dict = Metrics(metrics_list=metrics_list).evaluate(y_true=y, y_pred=y)
        return

if __name__=='__main__':
    unittest.main()