import unittest
import os
import sys
import shutil

sys.path.append('../')

from MASTML import MASTMLDriver

class TestMASTML(unittest.TestCase):

    def setUp(self):
        self.configfile = 'test_unittest_fullrun.conf'
        self.folders = list()
        return

    def test_runMASTML(self):
        MASTMLDriver(configfile=self.configfile).run_MASTML()
        self.folders.append('full_run')
        return

    def tearDown(self):
        for f in self.folders:
            shutil.rmtree(f)
        return

if __name__=='__main__':
    unittest.main()
