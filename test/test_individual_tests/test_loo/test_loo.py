import unittest
from nose import SkipTest
import os
import subprocess
from MASTML import MASTMLDriver
from test.test_helper import TestHelper

class TestLOO(TestHelper):
    def test_loo(self):
        tname="loo"
        self.run_command_mastml(tname)
        rmse=self.get_readme_line(tname, "avg_rmse")
        self.assertEqual(rmse, "avg_rmse: 292.895")
        clist = list()
        clist.append("README")
        clist.append("loo_results.png")
        clist.append("loo_results.ipynb")
        clist.append("loo_results.pickle")
        clist.append("loo_results_data_loo_prediction.csv")
        clist.append("loo_test_data.csv")
        self.compare_contents(tname, clist)
        self.remove_output(tname)
        return
    
