import unittest
from nose import SkipTest
import os
import subprocess
from MASTML import MASTMLDriver
from test.test_helper import TestHelper

class TestPredVsFeat(TestHelper):
    def test_predvsfeat(self):
        tname="predvsfeat"
        self.run_command_mastml(tname)
        clist = list()
        clist.append("README")
        clist.append("model.pickle")
        clist.append("feature_plot_group_A")
        clist.append("feature_plot_group_B")
        clist.append("feature_plot_group_C")
        clist.append("feature_plot_group_D")
        clist.append("feature_plot_group_None")
        clist.append("Initial")
        self.compare_contents(tname, clist)
        self.remove_output(tname)
        return
