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
    def test_dependencies(self): #use setup script or make more sophisticated later
        #raise SkipTest
        deplist=list()
        deplist.append("scikit-learn") #scikit-learn, imported as sklearn
        deplist.append("sklearn-deap")    #sklearn-deap, for evolutionary GA
        deplist.append("deap") #for evolutionary GA
        deplist.append("matplotlib") #for plotting
        deplist.append("PeakUtils") #for peak finding
        deplist.append("pymongo") #for mongoDB
        mydproc = subprocess.Popen("pip list", shell=True,
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        mydproc.wait()
        mydproc_result = mydproc.communicate()[0].decode("utf-8")
        mydeps = mydproc_result.split("\n")
        depnames=list()
        with open("environment_package_list","w") as pfile:
            for mydep in mydeps:
                mydep = mydep.strip()
                if len(mydep) > 0:
                    pfile.write("%s\n" % mydep)
                    depnames.append(mydep.split()[0])
        for dep in deplist:
            self.assertIn(dep,depnames)
        return
    def test_FullFit(self):
        #raise SkipTest
        wdir = os.path.join(tdir, "FullFit")
        os.chdir(wdir)
        self.run_command()
        return
    def test_KFoldCV(self):
        #raise SkipTest
        wdir = os.path.join(tdir, "KFoldCV")
        os.chdir(wdir)
        self.run_command()
        return
    def test_LeaveOneOutCV(self):
        #raise SkipTest
        wdir = os.path.join(tdir, "LeaveOneOutCV")
        os.chdir(wdir)
        self.run_command()
        return
    def test_LeaveOutGroupCV(self):
        #raise SkipTest
        wdir = os.path.join(tdir, "LeaveOutGroupCV")
        os.chdir(wdir)
        self.run_command()
        return
    def test_LeaveOutPercentCV(self):
        #raise SkipTest
        wdir = os.path.join(tdir, "LeaveOutPercentCV")
        os.chdir(wdir)
        self.run_command()
        return

    def run_command(self, verbose=1):
        mytproc = subprocess.Popen("python ../../AllTests.py", shell=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE)
        mytproc.wait()
        mytproc_result = mytproc.communicate()[0]
        if verbose > 0:
            print(mytproc_result)
        return mytproc_result

        
