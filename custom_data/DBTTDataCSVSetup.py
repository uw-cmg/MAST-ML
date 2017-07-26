#!/usr/bin/env python
###################
# Data import and export for the dbtt database
# Tam Mayeshiba 2017-02-03, modified 2017-07-21 to use pandas exclusively
#
# This script is intended to be run on a local computer in order to
# generate .csv files.
#
# Prerequisites:
# 1. Must have starting import csv files available.
#
###################
import numpy as np
import os
import sys
import traceback
import subprocess
import mongo_data.data_utilities.percent_converter as pconv
import time
from custom_features import cf_help
from DataOperations import DataParser
from FeatureOperations import FeatureIO, FeatureNormalization
import pandas as pd

class DBTTDataCSVSetup():
    def __init__(self, save_path="",
                    import_path="",
                    *args,
                    **kwargs):
        self.save_path = save_path
        self.import_path = import_path
        self.db = None
        self.dataframe = None
        self.init_names = dict()
        self.init_dfs = dict()
        self.dfs=dict()
        return

    def run(self):
        self.set_up()
        print('Creating expt ivar')
        self.csv_expt_ivar('expt_ivar')
        print('Creating cd2 ivar')
        self.csv_cd_ivar('cd2_ivar')
        print('Creating cd2 lwr')
        self.csv_cd_lwr('cd2_lwr')
        print('Creating expt atr2')
        self.csv_expt_atr2('expt_atr2')
        print('Creating standard lwr')
        self.csv_standard_lwr('standard_lwr')
        #for cname in self.dfs.keys():
        #    self.csv_add_features(cname)
        return

    def set_up(self):
        self.set_up_init_names()
        self.set_up_init_dfs()
        self.set_up_dfs()
        return

    def set_up_init_names(self):
        self.init_names=dict()
        self.init_names["alloys"] = "alloy_properties.csv"
        self.init_names["cd_ivar_2017"]="CD_IVAR_Hardening_2017-1_with_ivar_columns_reduced.csv"
        #self.init_names["cd_ivar_2016"]="CD_IVAR_Hardening_clean_2016.csv"
        self.init_names["cd_lwr_2017"]="lwr_cd_2017_reduced_for_import.csv"
        self.init_names["ucsb_ivar_and_ivarplus"]="ucsb_ivar_and_ivarplus.csv"
        self.init_names["cd_lwr_2016_bynum"]="CDTemp_CD_lwr_2016_raw.csv"
        self.init_names["atr2_2016"]="atr2_data.csv"
        return

    def set_up_init_dfs(self):
        for iname in self.init_names.keys():
            self.init_dfs[iname] = pd.read_csv(os.path.join(self.import_path, self.init_names[iname]))
        return

    def set_up_dfs(self):
        self.dfs['expt_ivar'] = self.init_dfs['ucsb_ivar_and_ivarplus'].copy()
        self.dfs['cd2_ivar'] = self.init_dfs['cd_ivar_2017'].copy()
        self.dfs['cd2_lwr'] = self.init_dfs['cd_lwr_2017'].copy()
        self.dfs['standard_lwr'] = ""
        self.dfs['expt_atr2'] = self.init_dfs['atr2_2016'].copy()
        return

    def csv_expt_ivar(self, cname):
        self.clean_expt_ivar(cname)
        self.add_standard_fields(cname)
        self.export_spreadsheet(cname)
        return

    def csv_cd_ivar(self, cname):
        self.dfs[cname].rename(columns={"CD_delta_sigma_y_MPa":"delta_sigma_y_MPa"}, inplace=True)
        self.clean_cd_ivar(cname) 
        self.add_standard_fields(cname)
        self.export_spreadsheet(cname)
        return

    def csv_cd_lwr(self, cname):
        self.dfs[cname].rename(columns={"CD_delta_sigma_y_MPa":"delta_sigma_y_MPa"}, inplace=True)
        self.clean_lwr(cname)
        if not "temperature_C" in self.dfs[cname].columns:
            print("Adding temperature column")
            self.dfs[cname]['temperature_C'] = 290.0
        self.add_standard_fields(cname)
        self.export_spreadsheet(cname)
        return

    def csv_expt_atr2(self, cname):
        self.dfs[cname].rename(columns={"alloy name": "Alloy"}, inplace=True)
        self.standardize_alloy_names(cname, verbose=0)
        self.dfs[cname]['dataset'] = "ATR2"
        self.dfs[cname]['time_sec'] = self.dfs[cname]['fluence_n_cm2'] / self.dfs[cname]['flux_n_cm2_sec']
        self.add_standard_fields(cname)
        self.export_spreadsheet(cname)
        return

    def csv_standard_lwr(self, cname):
        self.create_standard_conditions(cname,
                        ref_flux=3e10, temp=290, min_sec=3e6, max_sec=5e9)
        self.export_spreadsheet(cname) 
        return

    def clean_expt_ivar(self, cname, verbose=1):
        #remove LO
        self.dfs[cname].drop(self.dfs[cname][self.dfs[cname].Alloy == 'LO'].index, inplace=True)
        #self.dfs[cname].reset_index(drop=True, inplace=True)
        #print(self.dfs[cname])
        #
        #update some experimental temperatures
        self.update_experimental_temperatures(cname) ##MAYBE MOVE ABOVE CLOSE DUPS
        #
        #remove duplicates
        duplicates = self.get_true_duplicates_to_remove(cname)
        self.dfs[cname].drop(self.dfs[cname][duplicates].index, inplace=True)
        #
        #remove close duplicates
        close_duplicates = self.get_close_duplicates_to_remove(cname)
        self.dfs[cname].drop(self.dfs[cname][close_duplicates].index, inplace=True)
        #print(self.dfs[cname])
        return

    def get_true_duplicates_to_remove(self, cname):
        dup_criteria = list()
        dup_criteria.append("Alloy")
        dup_criteria.append("flux_n_cm2_sec")
        dup_criteria.append("fluence_n_cm2")
        dup_criteria.append("temperature_C")
        dup_criteria.append("delta_sigma_y_MPa")
        duplicates = self.dfs[cname].duplicated(subset = dup_criteria, keep='first')
        return duplicates
    
    def get_close_duplicates_to_remove(self, cname):
        """Where delta_sigma_y differs, remove the duplicate where
            conditions are least like the rest of the set.
            For eight such pairs, this happens to mean removing the
            smaller delta_sigma_y.
        """
        dup_criteria = list()
        dup_criteria.append("Alloy")
        dup_criteria.append("flux_n_cm2_sec")
        dup_criteria.append("fluence_n_cm2")
        dup_criteria.append("temperature_C")
        duplicates = self.dfs[cname].duplicated(subset = dup_criteria, keep=False)
        #print(duplicates[duplicates==True])
        close_dup_df = self.dfs[cname][duplicates]
        #print(close_dup_df)
        for inum in close_dup_df.index:
            i_item = close_dup_df[close_dup_df.index == inum]
            for jnum in close_dup_df.index:
                if inum == jnum:
                    continue
                j_item = close_dup_df[close_dup_df.index == jnum]
                if not(i_item.Alloy.item() == j_item.Alloy.item()):
                    continue
                if not(float(i_item.flux_n_cm2_sec) == float(j_item.flux_n_cm2_sec)):
                    continue
                if not(float(i_item.fluence_n_cm2) == float(j_item.fluence_n_cm2)):
                    continue
                if not(float(i_item.temperature_C) == float(j_item.temperature_C)):
                    continue
                if float(i_item.delta_sigma_y_MPa) > float(j_item.delta_sigma_y_MPa):
                    duplicates.set_value(inum, False)
                else:
                    duplicates.set_value(jnum, False)
        #print("filtered")
        #print(close_dup_df[duplicates== True])
        return duplicates

    def update_experimental_temperatures(self, cname, verbose=1):
        """Update temperatures for a handful of experimental points
            whose reported temperature in the original data sheet 
            for ucsb ivar plus were incorrect.
                Alloys CM6, LC, LD, LG, LH, LI at fluence of 1.10e21 n/cm2, 
                flux of 2.30e14 n/cm2/sec should all be at 
                Temperature = 320 degrees C instead of 290 degrees C.
                Their CD temperature, however, remains at 290. 
        """ 
        id_list=list()
        orig_temp_list=list()
        flux=2.30e14 #n/cm2/sec
        fluence=1.10e21 #n/cm2
        newtemp=320 #degrees C
        alloy_list=['CM6','LC','LD','LG','LH','LI']
        df = self.dfs[cname]
        to_update = df[(df.flux_n_cm2_sec == flux) & 
                        (df.fluence_n_cm2 == fluence) &
                        (df.Alloy.isin(alloy_list))]
        #print(to_update)
        self.dfs[cname].set_value(to_update.index, 'temperature_C', 320.0)
        #to_update = df[(df.flux_n_cm2_sec == flux) & 
        #                (df.fluence_n_cm2 == fluence) &
        #                (df.Alloy.isin(alloy_list))]
        #print(to_update)
        return

    def add_standard_fields(self, cname, verbose=0):
        """Add fields that are standard to most analysis
            Note that pandas indexing CHANGES after merges
        """
        self.add_alloy_fields(cname, verbose=0)
        self.add_atomic_percent_field(cname, verbose=0)
        
        self.add_log10_of_a_field(cname,"fluence_n_cm2")
        self.add_log10_of_a_field(cname,"flux_n_cm2_sec")

        for pval in [0.1,0.2,0.26]:
            self.add_generic_effective_fluence_field(cname, ref_flux=3e10, pval=pval, verbose=0)
        #TTM ParamOptGA can now add the appropriate effective fluence field 20170518
        self.add_normalization(cname, verbose)
        return

    def export_spreadsheet(self, cname):
        if not os.path.isdir(self.save_path):
            os.mkdir(self.save_path)
        self.dfs[cname].to_csv(os.path.join(self.save_path, "%s.csv" % cname))
        return

    def add_log10_of_a_field(self, cname, colname):
        self.dfs[cname]["log(%s)" % colname] = np.log10(self.dfs[cname][colname])
        #print(self.dfs[cname])
        return

    def add_generic_effective_fluence_field(self, cname, ref_flux=3e10, pval=1, verbose=0):
        """
            Calculated by fluence*(ref_flux/flux)^p 
            IVAR has ref_flux of 3e11 n/cm^2/sec (Odette 2005)
            However, LWR conditions have lower flux, so use 3e10 n/cm^2/sec.
            Args:
                ref_flux <float>: reference flux in n/cm^2/sec
                pval <float>: p value
            Also adds a log10 field
            G.R. Odette, T. Yamamoto, and D. Klingensmith, 
            Phil. Mag. 85, 779-797 (2005)
            With:
                flux <float> in n/cm2/sec
                fluence <float> in n/cm2
                ref_flux <float>: reference flux in n/cm2/sec
                pval <float>: p-value for exponent.
            Effective fluence takes the form:
            effective fluence ~= fluence * (ref_flux/flux)^p
            or equivalently:
            flux*effective time ~= flux*time * (ref_flux/flux)^p
        """
        pvalstr = "%i" % (pval*100.0)
        newfield = "eff fl 100p=%s" % pvalstr
        df = self.dfs[cname]
        self.dfs[cname][newfield] = df.fluence_n_cm2 * np.power((ref_flux/df.flux_n_cm2_sec), pval)
        self.add_log10_of_a_field(cname, newfield)
        if (verbose > 0):
            print(self.dfs[cname])
        return


    def add_alloy_fields(self, cname, verbose=0):
        """Add alloy number, product ID, and weight percents
        """
        acols = list()
        acols.append("Alloy")
        acols.append("alloy_number")
        acols.append("product_id")
        for elem in ["Cu","Ni","Mn","P","Si","C"]:
            acols.append("wt_percent_%s" % elem)
        self.dfs[cname] = self.dfs[cname].merge(self.init_dfs['alloys'][acols], on=['Alloy'])
        self.dfs[cname] = self.dfs[cname].fillna(0.0) #fill NaN in wt_percent_C column with zeros
        if verbose > 0:
            print(self.dfs[cname])
            print(self.dfs[cname][self.dfs[cname].isnull().any(axis=1)])
        return

    def add_atomic_percent_field(self, cname, verbose=0):
        elemlist=["Cu","Ni","Mn","P","Si","C"]
        for elem in elemlist:
            self.dfs[cname]["at_percent_%s" % elem] = 0.0 #float column
        for inum in self.dfs[cname].index:
            compstr=""
            for elem in elemlist:
                elem_wt = self.dfs[cname].get_value(inum, "wt_percent_%s" % elem)
                compstr = compstr + "%s %s," % (elem, elem_wt)
            compstr = compstr[:-1] # remove last comma
            outdict = pconv.main(compstr,'weight',verbose)
            for elem in elemlist:
                self.dfs[cname].set_value(inum, "at_percent_%s" % elem, float(outdict[elem]['perc_out']))
        if verbose > 0:
            print(self.dfs[cname])
        return

    def clean_cd_ivar(self, cname, verbose=1):
        #remove alloys with no hardness change and/or no data
        for alloy in ["CM1","CM2","CM8","LO"]: #also CM14, CM29?
            self.dfs[cname].drop(self.dfs[cname][self.dfs[cname].Alloy == alloy].index, inplace=True)
        #
        #remove duplicates
        duplicates = self.get_true_duplicates_to_remove(cname)
        self.dfs[cname].drop(self.dfs[cname][duplicates].index, inplace=True)
        #
        #[id_list, reason_list] = dclean.flag_bad_cd1_points(self.db, cname)
        return
    
    def standardize_alloy_names(self, cname, verbose=1):
        for inum in self.dfs[cname].index:
            ialloy = self.dfs[cname].get_value(inum, 'Alloy')
            alias_matches = self.init_dfs['alloys'][self.init_dfs['alloys'].alias_1 == ialloy]
            for mnum in alias_matches.index:
                standard_name = self.init_dfs['alloys'].get_value(mnum, 'Alloy')
                if (verbose > 0):
                    print("replacing %s with %s" % (ialloy, standard_name))
                self.dfs[cname].set_value(inum, 'Alloy', standard_name)
        return

    def standardize_flux_and_fluence(self, cname, verbose=1):
        self.dfs[cname]['flux_n_cm2_sec'] = self.dfs[cname]['flux_n_m2_sec'] / 100.0 / 100.0
        self.dfs[cname]['fluence_n_cm2'] = self.dfs[cname]['fluence_n_m2'] / 100.0 / 100.0
        return

    def clean_lwr(self, cname, verbose=0):
        # Standardize alloy names
        self.standardize_alloy_names(cname, verbose)
        # Use cm2 for flux and fluence, not m2
        self.standardize_flux_and_fluence(cname, verbose)
        # Drop empty flux and fluences
        self.dfs[cname].dropna(axis=0, subset=['flux_n_cm2_sec','fluence_n_cm2'], inplace=True)
        # Drop empty delta_sigma_y_MPa values
        self.dfs[cname].dropna(axis=0, subset=['delta_sigma_y_MPa'], inplace=True)
        #print(self.dfs[cname])
        if verbose > 0:
            print(self.dfs[cname][self.dfs[cname].isnull().any(axis=1)])
        # Remove short times (below 30000000 second)
        self.dfs[cname].drop(self.dfs[cname][self.dfs[cname].time_sec < 30e6].index, inplace=True)
        #print(self.dfs[cname])
        return


    def create_standard_conditions(self, cname, ref_flux=3e10, temp=290, min_sec=3e6, max_sec=5e9, clist=list(), verbose=0):
        #ref_flux in n/cm2/sec
        second_range = np.logspace(np.log10(min_sec), np.log10(max_sec), 50)
        alloys = self.init_dfs['alloys'].Alloy.tolist()
        index_size = len(alloys) * len(second_range)
        newdf = pd.DataFrame(index=range(0, index_size), columns=['Alloy','time_sec'])
        ict=0
        for alloy in alloys:
            for time_sec in second_range:
                newdf.set_value(ict, 'Alloy', alloy)
                newdf.set_value(ict, 'time_sec', time_sec)
                ict = ict + 1
        newdf['fluence_n_cm2'] = ref_flux * newdf.time_sec
        newdf['flux_n_cm2_sec'] = ref_flux
        newdf['temperature_C'] = temp
        self.dfs[cname] = newdf
        self.add_standard_fields(cname, verbose=0)
        return
    
    def add_normalization(self, cname, verbose=0):
        df = self.dfs[cname]
        norm_dict = dict()
        norm_dict['log(fluence_n_cm2)']=dict()
        norm_dict['log(fluence_n_cm2)']['smin'] = 17
        norm_dict['log(fluence_n_cm2)']['smax'] = 25
        norm_dict['log(flux_n_cm2_sec)']=dict()
        norm_dict['log(flux_n_cm2_sec)']['smin'] = 10
        norm_dict['log(flux_n_cm2_sec)']['smax'] = 15
        norm_dict['log(eff fl 100p=10)']=dict()
        norm_dict['log(eff fl 100p=10)']['smin'] = 17
        norm_dict['log(eff fl 100p=10)']['smax'] = 25
        norm_dict['log(eff fl 100p=20)']=dict()
        norm_dict['log(eff fl 100p=20)']['smin'] = 17
        norm_dict['log(eff fl 100p=20)']['smax'] = 25
        norm_dict['log(eff fl 100p=26)']=dict()
        norm_dict['log(eff fl 100p=26)']['smin'] = 17
        norm_dict['log(eff fl 100p=26)']['smax'] = 25
        norm_dict['temperature_C']=dict()
        norm_dict['temperature_C']['smin'] = 270
        norm_dict['temperature_C']['smax'] = 320
        for elem in ["P","C","Cu","Ni","Mn","Si"]:
            norm_dict["at_percent_%s" % elem] = dict()
            norm_dict["at_percent_%s" % elem]['smin'] = 0.0
            norm_dict["at_percent_%s" % elem]['smax'] = 1.717 #max Mn atomic percent
        for nkey in norm_dict.keys():
            fnorm = FeatureNormalization(df)
            scaled_feature = fnorm.minmax_scale_single_feature(nkey,
                                smin=norm_dict[nkey]['smin'], 
                                smax=norm_dict[nkey]['smax'])
            fio = FeatureIO(df)
            df = fio.add_custom_features(["N(%s)" % nkey],scaled_feature)
        self.dfs[cname] = df
        return

    def csv_add_features(self, csvsrc, csvdest):
        afm_dict=dict()
        param_dict=dict()
        #E900 column
        e900_dict = dict()
        for elem in ['P','Ni','Cu','Mn']: #Si, C not used in e900
            e900_dict['wt%s' % elem] = 'wt_percent_%s' % elem
        e900_dict['fluencestr'] = 'fluence_n_cm2'
        e900_dict['tempC'] = 'temperature_C'
        e900_dict['prod_ID'] = 'product_id'
        afm_dict['DBTT.E900'] = dict(e900_dict)
        param_dict['DBTT.E900'] = dict()
        #get_dataframe
        csv_dataparser = DataParser()
        csv_dataframe = csv_dataparser.import_data("%s.csv" % os.path.join(self.save_path, csvsrc))
        #add features
        for afm in afm_dict.keys():
            (feature_name, feature_data) = cf_help.get_custom_feature_data(class_method_str = afm,
                starting_dataframe = csv_dataframe,
                param_dict = dict(param_dict[afm]),
                addl_feature_method_kwargs = dict(afm_dict[afm]))
            fio = FeatureIO(csv_dataframe)
            csv_dataframe = fio.add_custom_features([feature_name],feature_data)
        #add log10 features
        log10_dict=dict()
        log10_dict['fluence_n_cm2'] = dict()
        log10_dict['flux_n_cm2_sec'] = dict()
        for lkey in log10_dict.keys():
            orig_data = csv_dataframe[lkey]
            log10_data = np.log10(orig_data)
            fio = FeatureIO(csv_dataframe)
            csv_dataframe = fio.add_custom_features(["log(%s)" % lkey], log10_data)
        #add normalizations
        norm_dict = dict()
        norm_dict['log(fluence_n_cm2)']=dict()
        norm_dict['log(fluence_n_cm2)']['smin'] = 17
        norm_dict['log(fluence_n_cm2)']['smax'] = 25
        norm_dict['log(flux_n_cm2_sec)']=dict()
        norm_dict['log(flux_n_cm2_sec)']['smin'] = 10
        norm_dict['log(flux_n_cm2_sec)']['smax'] = 15
        norm_dict['temperature_C']=dict()
        norm_dict['temperature_C']['smin'] = 270
        norm_dict['temperature_C']['smax'] = 320
        for elem in ["P","C","Cu","Ni","Mn","Si"]:
            norm_dict["at_percent_%s" % elem] = dict()
            norm_dict["at_percent_%s" % elem]['smin'] = 0.0
            norm_dict["at_percent_%s" % elem]['smax'] = 1.717 #max Mn atomic percent
        for nkey in norm_dict.keys():
            fnorm = FeatureNormalization(csv_dataframe)
            scaled_feature = fnorm.minmax_scale_single_feature(nkey,
                                smin=norm_dict[nkey]['smin'], 
                                smax=norm_dict[nkey]['smax'])
            fio = FeatureIO(csv_dataframe)
            csv_dataframe = fio.add_custom_features(["N(%s)" % nkey],scaled_feature)
        csv_dataframe.to_csv("%s.csv" % os.path.join(self.save_path, csvdest))
        return
if __name__ == "__main__":
    import_path = "../../../../data/DBTT_mongo/imports_201704"
    import_path = os.path.abspath(import_path)
    exportfolder = "exports_%s" %(time.strftime("%Y%m%d_%H%M%S"))
    save_path = "../../../../data/DBTT_mongo/%s" % exportfolder
    dbtt = DBTTData(save_path = save_path, import_path = import_path)
    dbtt.run()
