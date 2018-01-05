import unittest
from nose import SkipTest
import os
import subprocess
from MASTML import MASTMLDriver
from test.test_helper import TestHelper

class TestKFold(TestHelper):
    def test_kfold(self):
        tname="kfold"
        self.run_command_mastml(tname)
        rmse=self.get_readme_line(tname, "avg_fold_avg_rmses")
        self.assertEqual(rmse, "avg_fold_avg_rmses: 355.4589")
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
    
