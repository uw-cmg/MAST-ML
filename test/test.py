import unittest
from nose import SkipTest
import os
import subprocess
tdir=os.path.abspath(os.getcwd())

class TestBasic(unittest.TestCase):
    def setUp(self):
        pass
        return
    def tearDown(self):
        os.chdir(tdir)
        return

    def test_main(self):
        #raise SkipTest
        wdir = os.path.join(tdir, "main_test")
        os.chdir(wdir)
        self.run_command()
        return
    def run_command(self, verbose=1):
        mytproc = subprocess.Popen("nice -n 19 python ../../MASTML.py test.conf", shell=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE)
        mytproc.wait()
        mytproc_result = mytproc.communicate()[0]
        if verbose > 0:
            print(mytproc_result)
        return mytproc_result

        
