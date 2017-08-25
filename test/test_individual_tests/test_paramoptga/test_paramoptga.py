import unittest
from nose import SkipTest
import os
import subprocess
from MASTML import MASTMLDriver
from test.test_helper import TestHelper

class TestParamOptGA(TestHelper):
    def test_paramoptga(self):
        tname="paramoptga"
        try:
            self.remove_output(tname, remove=True)
        except OSError:
            pass
        self.run_command_mastml(tname)
        rmse=self.get_readme_line(tname, "Best 5-CV avg RMSE")
        self.assertEqual(rmse, "Best 5-CV avg RMSE: 163.687")
        subpath = self.get_subdirectory(tname)
        subdirs = os.listdir(subpath)
        gact=0
        for subdir in subdirs:
            if "GA_" in subdir:
                gact += 1
        self.assertEqual(gact, 2)
        a_opt = self.get_output_line(tname, "alpha","OPTIMIZED_PARAMS")
        self.assertEqual(a_opt, "model;alpha;1.0")
        g_opt = self.get_output_line(tname, "gamma","OPTIMIZED_PARAMS")
        self.assertEqual(g_opt, "model;gamma;0.001")
        return
    
