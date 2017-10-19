import unittest
from nose import SkipTest
import os
import subprocess
from MASTML import MASTMLDriver
from test.test_helper import TestHelper

class TestLeaveOutPercent(TestHelper):
    def test_leaveoutpercent(self):
        tname="loperc"
        self.run_command_mastml(tname)
        rmse=self.get_readme_line(tname, "avg_rmse")
        self.assertEqual(rmse, "avg_rmse: 297.9098")
        clist = list()
        clist.append("README")
        clist.append("best_worst_overlay.png")
        clist.append("best_worst_overlay.ipynb")
        clist.append("best_worst_overlay_data_Best_test.csv")
        clist.append("best_worst_overlay_data_Worst_test.csv")
        clist.append("best_worst_overlay.pickle")
        clist.append("best_and_worst_test_data.csv")
        self.compare_contents(tname, clist)
        self.remove_output(tname)
        return
    
