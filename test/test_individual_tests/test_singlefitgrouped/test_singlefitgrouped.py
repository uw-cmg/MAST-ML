import unittest
from nose import SkipTest
import os
import subprocess
from MASTML import MASTMLDriver
from test.test_helper import TestHelper

class TestSingleFitGrouped(TestHelper):
    def test_singlefitgrouped(self):
        self.run_command_mastml("singlefitgrouped")
        rmse=self.get_readme_line("singlefitgrouped", "SingleFitGrouped_KernelRidge0", "rmse")
        self.assertEqual(rmse, "rmse: 47.8267")
        clist = list()
        clist.append("README")
        clist.append("model.pickle")
        clist.append("output_data.csv")
        clist.append("single_fit.ipynb")
        clist.append("single_fit.pickle")
        clist.append("single_fit.png")
        clist.append("single_fit_data_predicted_vs_measured.csv")
        self.compare_contents("singlefitgrouped", "SingleFitGrouped_KernelRidge0",clist)
        return
    
    def test_only_matched_false(self):
        return

    def test_only_matched_true(self):
        return
