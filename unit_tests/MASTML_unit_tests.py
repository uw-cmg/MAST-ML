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
        self.configfile = 'example_input.conf'

        self.folders = list()
        return

    def mastml_basic_example(self):
        warnings.simplefilter("ignore")
        MASTMLDriver(configfile=self.configfile).run_MASTML()
        self.folders.append('example_results')
        return

    def tearDown(self):
        for f in self.folders:
            shutil.rmtree(f)
        return

if __name__=='__main__':
    unittest.main()
