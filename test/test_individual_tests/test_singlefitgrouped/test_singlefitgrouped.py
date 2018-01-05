import unittest
from nose import SkipTest
import os
import subprocess
from MASTML import MASTMLDriver
from test.test_helper import TestHelper

class TestSingleFitGrouped(TestHelper):
    def test_singlefitgrouped(self):
        tname="singlefitgrouped"
        self.run_command_mastml(tname)
        rmse=self.get_readme_line(tname, "rmse")
        self.assertEqual(rmse, "rmse: 47.8267")
        A_rmse = self.get_readme_line(tname, "A: rmse:")
        self.assertEqual(A_rmse.strip(), "A: rmse: 48.518")
        B_rmse = self.get_readme_line(tname, "B: rmse:")
        self.assertEqual(B_rmse.strip(), "B: rmse: 36.539")
        C_rmse = self.get_readme_line(tname, "C: rmse:")
        self.assertEqual(C_rmse.strip(), "C: rmse: 52.246")
        D_rmse = self.get_readme_line(tname, "D: rmse:")
        self.assertEqual(D_rmse.strip(), "D: rmse: 52.257")
        clist = list()
        clist.append("README")
        clist.append("model.pickle")
        clist.append("output_data.csv")
        clist.append("single_fit.ipynb")
        clist.append("single_fit.pickle")
        clist.append("single_fit.png")
        clist.append("single_fit_data_predicted_vs_measured.csv")
        clist.append("per_group_info")
        self.compare_contents(tname, clist)
        subpath = self.get_subdirectory(tname)
        pgpath = os.path.join(subpath, "per_group_info")
        plist = list()
        plist.append("per_group_info.ipynb")
        plist.append("per_group_info_data_All_others.csv")
        plist.append("per_group_info.pickle")
        plist.append("per_group_info.png")
        plist.append("per_group_info_data_C.csv")
        plist.append("per_group_info_data_D.csv")
        pgpathlist = os.listdir(pgpath)
        pgpathlist.sort() #sorts in place
        plist.sort()
        self.assertListEqual(pgpathlist, plist)
        self.remove_output(tname)
        return
    
    def test_filter(self):
        tname="filter"
        self.run_command_mastml(tname)
        f_rmse=self.get_readme_line(tname, "filtered_rmse")
        self.assertEqual(f_rmse, "filtered_rmse: 41.8855")
        Af_rmse=self.get_readme_line(tname, "A: filtered_rmse")
        self.assertEqual(Af_rmse, "A: filtered_rmse: 50.884")
        Bf_rmse=self.get_readme_line(tname, "B: filtered_rmse")
        self.assertEqual(Bf_rmse, "B: filtered_rmse: 32.118")
        Cf_rmse=self.get_readme_line(tname, "C: filtered_rmse")
        self.assertEqual(Cf_rmse, "C: filtered_rmse: 39.194")
        Df_rmse=self.get_readme_line(tname, "D: filtered_rmse")
        self.assertEqual(Df_rmse, "D: filtered_rmse: 43.504")
        self.remove_output(tname)
        return
    
    def test_only_matched_false(self):
        tname="alldata"
        self.run_command_mastml(tname)
        rmse=self.get_readme_line(tname, "rmse")
        self.assertEqual(rmse, "rmse: 123.3749")
        A_rmse = self.get_readme_line(tname, "A: rmse:")
        self.assertEqual(A_rmse.strip(), "A: rmse: 211.306")
        B_rmse = self.get_readme_line(tname, "B: rmse:")
        self.assertEqual(B_rmse.strip(), "B: rmse: 2.320")
        D_rmse = self.get_readme_line(tname, "D: rmse:")
        self.assertEqual(D_rmse.strip(), "D: rmse: 31.756")
        subpath = self.get_subdirectory(tname)
        pgpath = os.path.join(subpath, "per_group_info")
        plist = list()
        plist.append("per_group_info.ipynb")
        plist.append("per_group_info_data_All_others.csv")
        plist.append("per_group_info.pickle")
        plist.append("per_group_info.png")
        plist.append("per_group_info_data_D.csv")
        plist.append("per_group_info_data_A.csv")
        pgpathlist = os.listdir(pgpath)
        pgpathlist.sort() #sorts in place
        plist.sort()
        self.assertListEqual(pgpathlist, plist)
        self.remove_output(tname)
        return

    def test_only_matched_true(self):
        tname="match"
        self.run_command_mastml(tname)
        rmse=self.get_readme_line(tname, "rmse")
        self.assertEqual(rmse, "rmse: 160.7751")
        A_rmse = self.get_readme_line(tname, "A: rmse:")
        self.assertEqual(A_rmse.strip(), "A: rmse: 278.461")
        B_rmse = self.get_readme_line(tname, "B: rmse:")
        self.assertEqual(B_rmse.strip(), "B: rmse: 2.342")
        D_rmse = self.get_readme_line(tname, "D: rmse:")
        self.assertEqual(D_rmse.strip(), "D: rmse: 0.441")
        subpath = self.get_subdirectory(tname)
        pgpath = os.path.join(subpath, "per_group_info")
        plist = list()
        plist.append("per_group_info.ipynb")
        plist.append("per_group_info_data_All_others.csv")
        plist.append("per_group_info.pickle")
        plist.append("per_group_info.png")
        plist.append("per_group_info_data_B.csv")
        plist.append("per_group_info_data_A.csv")
        pgpathlist = os.listdir(pgpath)
        pgpathlist.sort() #sorts in place
        plist.sort()
        self.assertListEqual(pgpathlist, plist)
        self.remove_output(tname)
        return
