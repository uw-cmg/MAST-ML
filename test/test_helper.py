import unittest
from nose import SkipTest
import os
from MASTML import MASTMLDriver
import MASTML
import shutil
tdir=os.path.abspath(os.getcwd())

class TestHelper(unittest.TestCase):
    def setUp(self):
        os.chdir(tdir)
        self.test_folder = self.get_folder()
        mpath = os.path.join(self.test_folder, "MASTMLlog.log")
        if not os.path.isfile(mpath):
            with open(mpath,"w") as mfile:
                mfile.write("\n")
        os.chdir(self.test_folder)
        return

    def tearDown(self):
        pass
        return

    def remove_output(self, test_name, remove=True):
        rdir = os.path.join(self.test_folder, test_name)
        if remove is True:
            shutil.rmtree(rdir)
        os.chdir(tdir)
        return

    def run_command_mastml(self, test_name=""):
        fname = "%s.conf" % test_name
        mastml = MASTMLDriver(configfile=fname)
        mastml.run_MASTML() 
        return

    def get_subdirectory(self, test_name):
        testdir = test_name
        dirlist = os.listdir(test_name)
        for tdir in dirlist:
            tpath = os.path.join(testdir, tdir)
            if os.path.isdir(tpath):
                return tpath
        raise ValueError("No subdirectory found in %s." % testdir)
        return

    def get_readme_line(self, test_name, line_match):
        """Get a line from README
            Args:
                test_name (str): test name (should match test folder)
                line_match (str): string to match
            Returns:
                matching line if found
                None otherwise
        """
        subpath = self.get_subdirectory(test_name)
        with open(os.path.join(subpath, "README"),"r") as rfile:
            rlines = rfile.readlines()
        for rline in rlines:
            if line_match in rline:
                return rline.strip()
        return None

    def compare_contents(self, test_name, contentlist):
        """Compare directory contents
            Args:
                test_name (str): test name
                contentlist (list): list of directory contents
            Returns:
                Raises assertion error if directory contents and content list
                do not match.
        """
        subpath = self.get_subdirectory(test_name)
        contents=os.listdir(subpath)
        contents.sort()
        contentlist.sort()
        self.assertListEqual(contentlist, contents)
        return None

    def get_folder(self):
        """Get test folder
        """
        idstring = self.id()
        idsplit = idstring.split(".")
        relative_folder = str.join("/",idsplit[:-3])
        mastml_path = os.path.dirname(MASTML.__file__)
        test_folder = os.path.join(mastml_path, relative_folder)
        return test_folder
