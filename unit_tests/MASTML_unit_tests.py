import unittest
import os
import sys
import shutil
import warnings

# so we can find MASTML package
sys.path.append('../')

from MASTML import MASTMLDriver

class TestMASTML(unittest.TestCase):

    def setUp(self):

        self.folders = list()
        return

    def test_mastml_basic_example(self):
        configfile = 'example_input.conf'
        warnings.simplefilter('ignore')
        MASTMLDriver(configfile=configfile).run_MASTML()
        self.folders.append('example_results')
        return

    def test_mastml_full_run(self):
        configfile = 'full_run.conf'
        warnings.simplefilter('ignore')
        MASTMLDriver(configfile=configfile).run_MASTML()
        self.folders.append('full_run')
        return

    def tearDown(self):
        for f in self.folders:
            pass
            #shutil.rmtree(f)
        return

if __name__=='__main__':
    unittest.main()
