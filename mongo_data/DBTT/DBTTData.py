#!/usr/bin/env python
###################
# Data import and export for the dbtt database
# Tam Mayeshiba 2017-02-03
#
# This script is intended to be run on a local computer in order to
# generate .csv files.
#
# Prerequisites:
# 1. Must have mongodb installed and running.
#    Visit https://docs.mongodb.com/manual/administration/install-community/
# 2. Must have starting import csv files available.
#
###################
import numpy as np
import pymongo
import os
import sys
import traceback
import subprocess
import mongo_data.mongo_data_cleaning as mclean
import mongo_data.DBTT.DBTT_mongo_data_cleaning as dclean
import mongo_data.mongo_data_utilities as cas
import mongo_data.DBTT.DBTT_mongo_data_utilities as mcas
import mongo_data.DBTT.data_verification as dver
import mongo_data.DBTT.alloy_property_utilities as apu
import time
from pymongo import MongoClient
from bson.objectid import ObjectId
import mongo_data.mongo_utilities as mongoutil
from custom_features import cf_help
from DataOperations import DataParser
from FeatureOperations import FeatureIO, FeatureNormalization

class DBTTData():
    def __init__(self, save_path="",
                    import_path="",
                    *args,
                    **kwargs):
        self.save_path = save_path
        self.import_path = import_path
        self.db = None
        self.dataframe = None
        self.initial_collections = dict()
        return

    def run(self):
        self.set_up()
        self.mongo_expt_ivar()
        self.mongo_cd_ivar()
        self.mongo_cd_lwr()
        self.mongo_expt_atr2()
        self.mongo_standard_lwr()
        self.add_models_and_scaling()
        return

    def add_models_and_scaling(self):
        self.csv_add_features("cd2_ivar", "cd2_ivar_with_models_and_scaled")
        self.csv_add_features("cd2_lwr", "cd2_lwr_with_models_and_scaled")
        self.csv_add_features("standard_lwr","standard_lwr_with_models_and_scaled")
        self.csv_add_features("expt_ivar", "expt_ivar_with_models_and_scaled")
        self.csv_add_features("expt_atr2", "expt_atr2_with_models_and_scaled")
        return

    def set_up(self):
        self.set_up_mongo_db()
        self.set_up_initial_collections()
        return

    def set_up_mongo_db(self):
        db_base="dbtt"
        client = mongoutil.get_mongo_client()
        dbname = mongoutil.get_unique_name(client, db_base)
        self.db = client[dbname]
        return

    def set_up_initial_collections(self):
        self.initial_collections=dict()
        self.initial_collections["alloys"] = "alloy_properties.csv"
        self.initial_collections["cd_ivar_2017"]="CD_IVAR_Hardening_2017-1_with_ivar_columns_reduced.csv"
        self.initial_collections["cd_ivar_2016"]="CD_IVAR_Hardening_clean_2016.csv"
        self.initial_collections["cd_lwr_2017"]="lwr_cd_2017_reduced_for_import.csv"
        self.initial_collections["ucsb_ivar_and_ivarplus"]="ucsb_ivar_and_ivarplus.csv"
        self.initial_collections["cd_lwr_2016_bynum"]="CDTemp_CD_lwr_2016_raw.csv"
        self.initial_collections["atr2_2016"]="atr2_data.csv"
        for cname in self.initial_collections.keys():
            mongoutil.import_collection(self.db, cname, self.import_path, 
                    self.initial_collections[cname])
        return

    def mongo_expt_ivar(self):
        self.clean_expt_ivar("ucsb_ivar_and_ivarplus")
        cas.transfer_ignore_records(self.db, "ucsb_ivar_and_ivarplus","expt_ivar_ignore")
        cas.export_spreadsheet(self.db, "expt_ivar_ignore", self.save_path)
        cas.transfer_nonignore_records(self.db, "ucsb_ivar_and_ivarplus","expt_ivar")
        self.add_standard_fields("expt_ivar")
        cas.remove_field(self.db, "expt_ivar","original_reported_temperature_C")
        cas.export_spreadsheet(self.db, "expt_ivar", self.save_path)
        return

    def mongo_cd_ivar(self):
        cas.rename_field(self.db,"cd_ivar_2017","CD_delta_sigma_y_MPa","delta_sigma_y_MPa")
        self.clean_cd_ivar("cd_ivar_2017") 
        cas.transfer_ignore_records(self.db, "cd_ivar_2017","cd2_ivar_ignore")
        cas.export_spreadsheet(self.db, "cd2_ivar_ignore", self.save_path)
        cas.transfer_nonignore_records(self.db, "cd_ivar_2017","cd2_ivar")
        self.add_standard_fields("cd2_ivar")
        cas.export_spreadsheet(self.db, "cd2_ivar", self.save_path)
        return

    def mongo_cd_lwr(self):
        self.clean_lwr("cd_lwr_2017")
        self.create_lwr("cd2_lwr", "cd_lwr_2017")
        cas.export_spreadsheet(self.db, "cd2_lwr", self.save_path)
        return

    def mongo_expt_atr2(self):
        cas.transfer_ignore_records(self.db, "atr2_2016","expt_atr2_ignore")
        cas.export_spreadsheet(self.db, "atr2_2016_ignore", self.save_path)
        cas.transfer_nonignore_records(self.db, "atr2_2016","expt_atr2")
        cas.rename_field(self.db,"expt_atr2","alloy name", "Alloy")
        dclean.standardize_alloy_names(self.db,"expt_atr2")
        cas.add_basic_field(self.db, "expt_atr2", "dataset", "ATR2")
        mcas.add_time_field(self.db, "expt_atr2")
        self.add_standard_fields("expt_atr2")
        cas.export_spreadsheet(self.db, "expt_atr2", self.save_path)
        return

    def mongo_standard_lwr(self):
        self.create_standard_conditions("standard_lwr",
                        ref_flux=3e10, temp=290, min_sec=3e6, max_sec=5e9)
        cas.export_spreadsheet(self.db, "standard_lwr", self.save_path)
        return

    def clean_expt_ivar(self, cname, verbose=1):
        [id_list, reason_list] = dclean.get_alloy_removal_ids(self.db, cname, [41])
        mclean.flag_for_ignore(self.db, cname, id_list, reason_list)
        print(len(id_list))
        [id_list, reason_list] = dclean.get_duplicate_ids_to_remove(self.db, cname)
        mclean.flag_for_ignore(self.db, cname, id_list, reason_list)
        print(len(id_list))
        dclean.update_experimental_temperatures(self.db, cname)
        return

    def add_standard_fields(self, cname, verbose=0):
        """Add fields that are standard to most analysis
        """
        mcas.add_alloy_number_field(self.db, cname, verbose=0)
        mcas.add_product_id_field(self.db, cname, verbose=0)
        mcas.add_weight_percent_field(self.db, cname, verbose=0)
        mcas.add_atomic_percent_field(self.db, cname, verbose=0)
        #cas.add_log10_of_a_field(db, cname,"fluence_n_cm2")
        #cas.add_log10_of_a_field(db, cname,"flux_n_cm2_sec")
        #TTM ParamOptGA can now add the appropriate effective fluence field 20170518
        #for pval in np.arange(0.0,1.01,0.01):
        #    pvalstr = "%i" % (100*pval)
        #    cas.add_generic_effective_fluence_field(db, cname, 3e10, pval)
        return

    def clean_cd_ivar(self, cname, verbose=1):
        [id_list, reason_list] = dclean.get_alloy_removal_ids(self.db, cname, 
                                    [1,2,8,41])
        #                            [41,1,2,8,14,29])
        mclean.flag_for_ignore(self.db, cname, id_list, reason_list)
        print(len(id_list))
        [id_list, reason_list] = dclean.get_duplicate_ids_to_remove(self.db, cname)
        mclean.flag_for_ignore(self.db, cname, id_list, reason_list)
        print(len(id_list))
        [id_list, reason_list] = dclean.flag_bad_cd1_points(self.db, cname)
        mclean.flag_for_ignore(self.db, cname, id_list, reason_list)
        print(len(id_list))
        return


    def clean_lwr(self, cname, verbose=1):
        dclean.standardize_alloy_names(self.db, cname)
        [id_list, reason_list] = dclean.get_alloy_removal_ids(self.db, cname,
                    [1,2,8,41])
        mclean.flag_for_ignore(self.db, cname, id_list, reason_list)
        print(len(id_list))
        [id_list, reason_list] = dclean.get_empty_flux_or_fluence_removal_ids(self.db, cname)
        mclean.flag_for_ignore(self.db, cname, id_list, reason_list)
        print(len(id_list))
        [id_list, reason_list] = dclean.get_short_time_removal_ids(self.db,cname, 3e6)
        mclean.flag_for_ignore(self.db, cname, id_list, reason_list)
        print(len(id_list))
        [id_list, reason_list] = mclean.get_field_condition_to_remove(self.db,cname,
                                    "CD_delta_sigma_y_MPa","")
        mclean.flag_for_ignore(self.db, cname, id_list, reason_list)
        print(len(id_list))
        return


    def create_lwr(self, cname, fromcname, verbose=1):
        """Create LWR condition spreadsheet
        """
        cas.transfer_ignore_records(self.db, fromcname, "%s_ignore" % cname, verbose)
        cas.export_spreadsheet(self.db, "%s_ignore" % cname, self.save_path)
        cas.transfer_nonignore_records(self.db, fromcname, cname, verbose)
        #Additional cleaning. Flux and fluence must be present for all records.
        dclean.standardize_flux_and_fluence(self.db, cname)
        cas.rename_field(self.db, cname, "CD_delta_sigma_y_MPa", "delta_sigma_y_MPa")
        if not "temperature_C" in cas.list_all_fields(self.db, cname):
            cas.add_basic_field(self.db, cname, "temperature_C", 290.0) # all at 290
        self.add_standard_fields(cname)
        return

    def reformat_lwr(db, cname, fromcname, verbose=1):
        """Reformat CD LWR 2016 where each record has a number of
            columns for each alloy number
        """
        raise NotImplementedError()
        alloy_numbers = apu.get_alloy_numbers(db)
        fields = cas.list_all_fields(db, fromcname)
        transferfields = list(fields)
        transferfields.remove("_id") # do not copy over ID from previous db
        for alloy_num in alloy_numbers:
            if str(alloy_num) in transferfields: #filter out alloy numbers
                transferfields.remove(str(alloy_num))
        records = db[fromcname].find()
        for record in records:
            for alloy_num in alloy_numbers:
                idict=dict()
                for tfield in transferfields:
                    idict[tfield] = record[tfield]
                try: 
                    dsyval = float(record["%i" % alloy_num])
                except (ValueError, KeyError): #might be Err!, blank, or not exist
                    continue
                idict["delta_sigma_y_MPa"] = dsyval
                alloy_name = apu.look_up_name_or_number(db,alloy_num,"number")
                idict["Alloy"] = alloy_name
                db[cname].insert_one(idict)
        return

    def create_standard_conditions(self, cname, ref_flux=3e10, temp=290, min_sec=3e6, max_sec=5e9, clist=list(), verbose=0):
        #ref_flux in n/cm2/sec
        second_range = np.logspace(np.log10(min_sec), np.log10(max_sec), 50)
        alloys = apu.get_alloy_names(self.db)
        for alloy in alloys:
            for time_sec in second_range:
                fluence = ref_flux * time_sec
                self.db[cname].insert_one({"Alloy": alloy,
                                    "time_sec": time_sec,
                                    "fluence_n_cm2": fluence,
                                    "flux_n_cm2_sec": ref_flux})
        cas.add_basic_field(self.db, cname, "temperature_C", temp)
        mcas.add_alloy_number_field(self.db, cname, verbose=0)
        mcas.add_product_id_field(self.db, cname, verbose=0)
        mcas.add_weight_percent_field(self.db, cname, verbose=0)
        mcas.add_atomic_percent_field(self.db, cname, verbose=0)
        #cas.add_log10_of_a_field(db, cname,"fluence_n_cm2")
        #cas.add_log10_of_a_field(db, cname,"flux_n_cm2_sec")
        ##TTM ParamOptGA will now add field
        ##for pval in np.arange(0.0,1.01,0.01):
        ##    pvalstr = "%i" % (100*pval)
        ##    mcas.add_generic_effective_fluence_field(db, cname, 3e10, pval)
        ##    cas.add_minmax_normalization_of_a_field(db, cname, 
        ##            "log(eff fl 100p=%s)" % pvalstr,
        ##            verbose=verbose, collectionlist = clist)
        #cas.add_minmax_normalization_of_a_field(db, cname, "temperature_C",
        #        setmin=270,setmax=320,
        #        verbose=verbose, collectionlist = clist)
        #cas.add_minmax_normalization_of_a_field(db, cname, "log(fluence_n_cm2)",
        #        setmin=17, setmax=25,
        #        verbose=verbose, collectionlist = clist)
        #cas.add_minmax_normalization_of_a_field(db, cname, "log(flux_n_cm2_sec)",
        #        setmin=10, setmax=15,
        #        verbose=verbose, collectionlist = clist)
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
