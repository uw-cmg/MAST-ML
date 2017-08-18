import unittest
from nose import SkipTest
import os
import subprocess
tdir=os.path.abspath(os.getcwd())
from MASTML import MASTMLDriver
import MASTML

class TestHelper(unittest.TestCase):
    def setUp(self):
        self.test_folder = self.get_folder()
        mpath = os.path.join(self.test_folder, "MASTMLlog.log")
        if not os.path.isfile(mpath):
            with open(mpath,"w") as mfile:
                mfile.write("\n")
        os.chdir(self.test_folder)
        return

    def tearDown(self):
        os.chdir(tdir)
        return

    def run_command_mastml(self, test_name=""):
        fname = "test_%s.conf" % test_name
        mastml = MASTMLDriver(configfile=fname)
        mastml.run_MASTML() 
        return

    def get_readme_line(self, test_name, subname, line_match):
        """Get a line from README
            Args:
                test_name (str): test name
                subname (str): subfolder name, like SingleFit_KernelRidge0
                line_match (str): string to match
            Returns:
                matching line if found
                None otherwise
        """
        testdir = "output_test_%s" % test_name
        with open(os.path.join(testdir, subname, "README"),"r") as rfile:
            rlines = rfile.readlines()
        for rline in rlines:
            if line_match in rline:
                return rline.strip()
        return None

    def compare_contents(self, test_name, subname, contentlist):
        """Compare directory contents
            Args:
                test_name (str): test name
                subname (str): subfolder name, like SingleFit_KernelRidge0
                contentlist (list): list of directory contents
            Returns:
                Raises assertion error if directory contents and content list
                do not match.
        """
        testdir = "output_test_%s" % test_name
        contents=os.listdir(os.path.join(testdir, subname))
        contents.sort()
        contentlist.sort()
        self.assertListEqual(contentlist, contents)
        return None

    def get_folder(self):
        """Get test folder
        """
        idstring = self.id()
        idsplit = idstring.split(".")
        relative_folder = str.join("/",idsplit[0:3])
        mastml_path = os.path.dirname(MASTML.__file__)
        test_folder = os.path.join(mastml_path, relative_folder)
        return test_folder
