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
        return

    def test_basic_example(self):
        configfile = 'example_input.conf'
        warnings.simplefilter('ignore')
        MASTMLDriver(configfile=configfile).run_MASTML()
        self.folders.append('results/example_results')
        return

    def test_full_run(self):
        configfile = 'full_run.conf'
        warnings.simplefilter('ignore')
        MASTMLDriver(configfile=configfile).run_MASTML()
        self.folders.append('results/full_run')
        return

    def nota_test_classifiers(self):
        configfile = 'classifiers.conf'
        warnings.simplefilter('ignore')
        MASTMLDriver(configfile=configfile).run_MASTML()
        self.folders.append('results/classifiers')
        return

    def tearDown(self):
        for f in self.folders:
            pass
            #shutil.rmtree(f)
        return

if __name__=='__main__':
    unittest.main()
