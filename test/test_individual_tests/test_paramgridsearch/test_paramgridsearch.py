import unittest
from nose import SkipTest
import os
import subprocess
from MASTML import MASTMLDriver
from test.test_helper import TestHelper

class TestParamGridSearch(TestHelper):
    def test_paramgridsearch(self):
        tname="paramgridsearch"
        self.run_command_mastml(tname)
        rmse=self.get_readme_line(tname, "4_0_0_0")
        self.assertEqual(rmse.split(",")[0], "4_0_0_0: 157.035")
        subpath = self.get_subdirectory(tname)
        subdirs = os.listdir(subpath)
        gridct=0
        for subdir in subdirs:
            if "indiv_" in subdir:
                gridct += 1
        self.assertEqual(gridct, 84)
        return
    
    def test_leaveout(self):
        tname="leaveout"
        self.run_command_mastml(tname)
        rmse=self.get_readme_line(tname, "4_0_1_1")
        self.assertEqual(rmse.split(",")[0], "4_0_1_1: 157.725")
        return
