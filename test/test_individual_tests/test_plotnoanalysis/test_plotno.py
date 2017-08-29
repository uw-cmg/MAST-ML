import unittest
from nose import SkipTest
import os
import subprocess
from MASTML import MASTMLDriver
from test.test_helper import TestHelper

class TestPlotNo(TestHelper):
    def test_plotno(self):
        tname="plotno"
        self.run_command_mastml(tname)
        clist = list()
        clist.append("README")
        for group in ['A','B','C','D','all']:
            clist.append("%s.ipynb" % group)
            clist.append("%s.pickle" % group)
            clist.append("%s.png" % group)
            clist.append("%s_data_Initial.csv" % group)
        clist.append("output_Initial.csv")
        self.compare_contents(tname, clist)
        self.remove_output(tname)
        return
