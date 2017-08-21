import unittest
from nose import SkipTest
import os
import subprocess
from MASTML import MASTMLDriver
from test.test_helper import TestHelper

class TestSingleFit(TestHelper):
    def test_singlefit(self):
        tname="singlefit"
        self.run_command_mastml(tname)
        rmse=self.get_readme_line(tname, "rmse")
        self.assertEqual(rmse, "rmse: 47.8267")
        clist = list()
        clist.append("README")
        clist.append("model.pickle")
        clist.append("output_data.csv")
        clist.append("single_fit.ipynb")
        clist.append("single_fit.pickle")
        clist.append("single_fit.png")
        clist.append("single_fit_data_predicted_vs_measured.csv")
        self.compare_contents(tname, clist)
        self.remove_output(tname)
        return
    
    def test_filter(self):
        tname = "filter"
        self.run_command_mastml(tname)
        rmse=self.get_readme_line(tname, "filtered_rmse")
        self.assertEqual(rmse, "filtered_rmse: 56.3935")
        clist = list()
        clist.append("README")
        clist.append("model.pickle")
        clist.append("output_data.csv")
        clist.append("output_data_filtered.csv")
        clist.append("single_fit.ipynb")
        clist.append("single_fit.pickle")
        clist.append("single_fit.png")
        clist.append("single_fit_data_predicted_vs_measured.csv")
        self.compare_contents(tname, clist)
        self.remove_output(tname)
        return
    
    def test_notarget(self):
        tname="notarget"
        self.run_command_mastml(tname)
        rmse=self.get_readme_line(tname, "rmse")
        self.assertEqual(rmse, None) #no statistics because no target data
        clist = list()
        clist.append("README")
        clist.append("model.pickle")
        clist.append("output_data.csv")
        self.compare_contents(tname, clist)
        self.remove_output(tname)
        return

