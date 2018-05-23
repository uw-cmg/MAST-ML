import unittest
import os
import sys
import shutil
import warnings
import pdb

# so we can find MASTML package
sys.path.append('../')

from MASTML import MASTMLDriver

class TestMASTML(unittest.TestCase):

    def setUp(self):
        self.folders = list()

    def test_full_run(self):
        configfile = 'full_run.conf'
        warnings.simplefilter('ignore')
        MASTMLDriver(configfile=configfile).run_MASTML()
        self.folders.append('results/full_run')

    def test_basic_example(self):
        configfile = 'example_input.conf'
        warnings.simplefilter('ignore')
        MASTMLDriver(configfile=configfile).run_MASTML()
        self.folders.append('results/example_results')

    def test_classifiers(self):
        configfile = 'classifiers.conf'
        warnings.simplefilter('ignore')
        MASTMLDriver(configfile=configfile).run_MASTML()
        self.folders.append('results/classifiers')

    def tearDown(self):
        for f in self.folders:
            pass
            # uncomment for no leftover folders:
            #shutil.rmtree(f)

if __name__=='__main__':
    unittest.main()
