import unittest
from nose import SkipTest
import os
import subprocess
from MASTML import MASTMLDriver
from test.test_helper import TestHelper

class TestLeaveOutGroup(TestHelper):
    def test_leaveoutgroup(self):
        tname="logroup"
        self.run_command_mastml(tname)
        rmse=self.get_readme_line(tname, "A: rmse")
        self.assertEqual(rmse, "A: rmse: 319.605")
        clist = list()
        clist.append("README")
        clist.append("A_test_data.csv")
        clist.append("B_test_data.csv")
        clist.append("C_test_data.csv")
        clist.append("D_test_data.csv")
        clist.append("leave_out_group.ipynb")
        clist.append("leave_out_group.png")
        clist.append("leave_out_group.pickle")
        clist.append("leave_out_group_data_predicted_rmse.csv")
        self.compare_contents(tname, clist)
        self.remove_output(tname)
        return
    
