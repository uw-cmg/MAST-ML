import unittest
from nose import SkipTest
import os
import subprocess
from MASTML import MASTMLDriver
from test.test_helper import TestHelper

class TestSingleFit(TestHelper):
    def test_singlefit(self):
        self.run_command_mastml("singlefit")
        rmse=self.get_readme_line("singlefit", "SingleFit_KernelRidge0", "rmse")
        self.assertEqual(rmse, "rmse: 47.8267")
        clist = list()
        clist.append("README")
        clist.append("model.pickle")
        clist.append("output_data.csv")
        clist.append("single_fit.ipynb")
        clist.append("single_fit.pickle")
        clist.append("single_fit.png")
        clist.append("single_fit_data_predicted_vs_measured.csv")
        self.compare_contents("singlefit", "SingleFit_KernelRidge0",clist)
        return
    
    def test_filter(self):
        self.run_command_mastml("filter")
        rmse=self.get_readme_line("filter", "SingleFit_filter_KernelRidge0", 
                        "filtered_rmse")
        self.assertEqual(rmse, "filtered_rmse: 56.3935")
        clist = list()
        clist.append("README")
        clist.append("model.pickle")
        clist.append("output_data.csv")
        clist.append("single_fit.ipynb")
        clist.append("single_fit.pickle")
        clist.append("single_fit.png")
        clist.append("single_fit_data_predicted_vs_measured.csv")
        self.compare_contents("singlefit", "SingleFit_KernelRidge0",clist)
        return

