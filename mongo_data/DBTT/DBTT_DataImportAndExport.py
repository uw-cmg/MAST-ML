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

def import_initial_collections(db, importpath):
    """Import initial collections for use in creating specialized collections.
    """
    cbasic=dict()
    cbasic["alloys"] = "alloy_properties.csv"
    cbasic["cd_ivar_2017"]="CD_IVAR_Hardening_2017-1_with_ivar_columns_reduced.csv"
    cbasic["cd_ivar_2016"]="CD_IVAR_Hardening_clean_2016.csv"
    cbasic["cd_lwr_2017"]="lwr_cd_2017_reduced_for_import.csv"
    cbasic["ucsb_ivar_and_ivarplus"]="ucsb_ivar_and_ivarplus.csv"
    cbasic["cd_lwr_2016_bynum"]="CDTemp_CD_lwr_2016_raw.csv"
    cbasic["atr2_2016"]="atr2_data.csv"
    for cname in cbasic.keys():
        mongoutil.import_collection(db, cname, importpath, cbasic[cname])
    return

def clean_expt_ivar(db, cname, verbose=1):
    [id_list, reason_list] = dclean.get_alloy_removal_ids(db, cname, [41])
    mclean.flag_for_ignore(db, cname, id_list, reason_list)
    print(len(id_list))
    [id_list, reason_list] = dclean.get_duplicate_ids_to_remove(db, cname)
    mclean.flag_for_ignore(db, cname, id_list, reason_list)
    print(len(id_list))
    dclean.update_experimental_temperatures(db, cname)
    return

def add_standard_fields(db, cname, verbose=0):
    """Add fields that are standard to most analysis
    """
    mcas.add_alloy_number_field(db, cname, verbose=0)
    mcas.add_atomic_percent_field(db, cname, verbose=0)
    cas.add_log10_of_a_field(db, cname,"fluence_n_cm2")
    cas.add_log10_of_a_field(db, cname,"flux_n_cm2_sec")
    #TTM ParamOptGA can now add the appropriate effective fluence field 20170518
    #for pval in np.arange(0.0,1.01,0.01):
    #    pvalstr = "%i" % (100*pval)
    #    cas.add_generic_effective_fluence_field(db, cname, 3e10, pval)
    return

def add_normalized_fields(db, cname, clist=list(), verbose=0):
    cas.add_minmax_normalization_of_a_field(db, cname, "log(fluence_n_cm2)",
            setmin=17, setmax=25, 
            verbose=verbose, collectionlist = clist) #fluences 1e17 to 1e25
    cas.add_minmax_normalization_of_a_field(db, cname, "log(flux_n_cm2_sec)",
            setmin=10, setmax=15,
            verbose=verbose, collectionlist = clist) #fluxes 7e10 to 2.3e14
    cas.add_minmax_normalization_of_a_field(db, cname, "temperature_C",
            setmin=270,setmax=320,
            verbose=verbose, collectionlist = clist)
    #TTM ParamOptGA will now normalize the eff fluence fields as needed
    #for pval in np.arange(0.0,1.01,0.01):
    #    pvalstr = "%i" % (100*pval)
    #    cas.add_minmax_normalization_of_a_field(db, cname, 
    #            "log(eff fl 100p=%s)" % pvalstr,
    #            verbose=verbose, collectionlist = clist)
    #cas.add_stddev_normalization_of_a_field(db, cname, "delta_sigma_y_MPa",
    #        verbose = verbose, collectionlist = clist)
    return


def clean_cd1_ivar(db, cname, verbose=1):
    [id_list, reason_list] = dclean.get_alloy_removal_ids(db, cname, 
                                [41,1,2,8,14,29])
    mclean.flag_for_ignore(db, cname, id_list, reason_list)
    print(len(id_list))
    [id_list, reason_list] = dclean.get_duplicate_ids_to_remove(db, cname)
    mclean.flag_for_ignore(db, cname, id_list, reason_list)
    print(len(id_list))
    [id_list, reason_list] = dclean.flag_bad_cd1_points(db, cname)
    mclean.flag_for_ignore(db, cname, id_list, reason_list)
    print(len(id_list))
    return


def clean_lwr(db, cname, verbose=1):
    dclean.standardize_alloy_names(db, cname)
    [id_list, reason_list] = dclean.get_alloy_removal_ids(db, cname,[1,2,8,41])
    mclean.flag_for_ignore(db, cname, id_list, reason_list)
    print(len(id_list))
    
    [id_list, reason_list] = dclean.get_empty_flux_or_fluence_removal_ids(db, cname)
    mclean.flag_for_ignore(db, cname, id_list, reason_list)
    print(len(id_list))
    
    [id_list, reason_list] = dclean.get_short_time_removal_ids(db,cname, 3e6)
    mclean.flag_for_ignore(db, cname, id_list, reason_list)
    print(len(id_list))
    
    [id_list, reason_list] = mclean.get_field_condition_to_remove(db,cname,
                                "CD_delta_sigma_y_MPa","")
    mclean.flag_for_ignore(db, cname, id_list, reason_list)
    print(len(id_list))
    return

def clean_cd1_lwr(db, cname):
    [id_list, reason_list] = dclean.get_alloy_removal_ids(db, cname,[14,29])
    mclean.flag_for_ignore(db, cname, id_list, reason_list)
    print(len(id_list))
    return

def create_lwr(db, cname, fromcname, exportpath, verbose=1):
    """Create LWR condition spreadsheet
    """
    cas.transfer_ignore_records(db, fromcname, "%s_ignore" % cname, verbose)
    cas.export_spreadsheet(db, "%s_ignore" % cname, exportpath)
    cas.transfer_nonignore_records(db, fromcname, cname, verbose)
    #Additional cleaning. Flux and fluence must be present for all records.
    dclean.standardize_flux_and_fluence(db, cname)
    cas.rename_field(db, cname, "CD_delta_sigma_y_MPa", "delta_sigma_y_MPa")
    if not "temperature_C" in cas.list_all_fields(db, cname):
        cas.add_basic_field(db, cname, "temperature_C", 290.0) # all at 290
    add_standard_fields(db, cname)
    return

def reformat_lwr(db, cname, fromcname, verbose=1):
    """Reformat CD LWR 2016 where each record has a number of
        columns for each alloy number
    """
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

def create_standard_conditions(db, cname, ref_flux=3e10, temp=290, min_sec=3e6, max_sec=5e9, clist=list(), verbose=0):
    #ref_flux in n/cm2/sec
    second_range = np.logspace(np.log10(min_sec), np.log10(max_sec), 50)
    alloys = apu.get_alloy_names(db)
    for alloy in alloys:
        for time_sec in second_range:
            fluence = ref_flux * time_sec
            db[cname].insert_one({"Alloy": alloy,
                                "time_sec": time_sec,
                                "fluence_n_cm2": fluence,
                                "flux_n_cm2_sec": ref_flux})
    cas.add_basic_field(db, cname, "temperature_C", temp)
    mcas.add_alloy_number_field(db, cname, verbose=0)
    mcas.add_atomic_percent_field(db, cname, verbose=0)
    cas.add_log10_of_a_field(db, cname,"fluence_n_cm2")
    cas.add_log10_of_a_field(db, cname,"flux_n_cm2_sec")
    #TTM ParamOptGA will now add field
    #for pval in np.arange(0.0,1.01,0.01):
    #    pvalstr = "%i" % (100*pval)
    #    mcas.add_generic_effective_fluence_field(db, cname, 3e10, pval)
    #    cas.add_minmax_normalization_of_a_field(db, cname, 
    #            "log(eff fl 100p=%s)" % pvalstr,
    #            verbose=verbose, collectionlist = clist)
    cas.add_minmax_normalization_of_a_field(db, cname, "temperature_C",
            setmin=270,setmax=290,
            verbose=verbose, collectionlist = clist)
    cas.add_minmax_normalization_of_a_field(db, cname, "log(fluence_n_cm2)",
            setmin=17, setmax=25,
            verbose=verbose, collectionlist = clist)
    cas.add_minmax_normalization_of_a_field(db, cname, "log(flux_n_cm2_sec)",
            setmin=10, setmax=15,
            verbose=verbose, collectionlist = clist)
    return

def main(importpath):
    #set up database
    dirpath = os.path.dirname(importpath)
    db_base="dbtt"
    client = mongoutil.get_mongo_client()
    dbname = mongoutil.get_unique_name(client, db_base)
    exportfolder = "data_exports_%s_%s" %(dbname,time.strftime("%Y%m%d_%H%M%S"))
    exportpath = os.path.join(dirpath, exportfolder)
    db = client[dbname]
    #import initial collections
    import_initial_collections(db, importpath)
    #create ancillary databases and spreadsheets
    ##Expt IVAR
    clean_expt_ivar(db, "ucsb_ivar_and_ivarplus")
    cas.transfer_ignore_records(db, "ucsb_ivar_and_ivarplus","expt_ivar_ignore")
    cas.export_spreadsheet(db, "expt_ivar_ignore", exportpath)
    cas.transfer_nonignore_records(db, "ucsb_ivar_and_ivarplus","expt_ivar")
    add_standard_fields(db, "expt_ivar")
    ##CD1 IVAR
    #cas.rename_field(db,"cd_ivar_2016","CD_delta_sigma_y_MPa", "delta_sigma_y_MPa")
    #clean_cd1_ivar(db, "cd_ivar_2016")
    #cas.transfer_ignore_records(db, "cd_ivar_2016","cd1_ivar_ignore")
    #cas.export_spreadsheet(db, "cd1_ivar_ignore", exportpath)
    #cas.transfer_nonignore_records(db, "cd_ivar_2016","cd1_ivar")
    #add_standard_fields(db, "cd1_ivar")
    ##CD2 IVAR
    cas.rename_field(db,"cd_ivar_2017","CD_delta_sigma_y_MPa","delta_sigma_y_MPa")
    clean_cd1_ivar(db, "cd_ivar_2017") 
    cas.transfer_ignore_records(db, "cd_ivar_2017","cd2_ivar_ignore")
    cas.export_spreadsheet(db, "cd2_ivar_ignore", exportpath)
    cas.transfer_nonignore_records(db, "cd_ivar_2017","cd2_ivar")
    add_standard_fields(db, "cd2_ivar")
    ##CD1 LWR
    #reformat_lwr(db, "cd_lwr_2016", "cd_lwr_2016_bynum")
    #clean_cd1_lwr(db, "cd_lwr_2016")
    #clean_lwr(db, "cd_lwr_2016")
    #create_lwr(db, "cd1_lwr", "cd_lwr_2016", exportpath)
    ##CD2 LWR
    clean_lwr(db, "cd_lwr_2017")
    create_lwr(db, "cd2_lwr", "cd_lwr_2017", exportpath)
    ##ATR2
    cas.transfer_ignore_records(db, "atr2_2016","expt_atr2_ignore")
    cas.export_spreadsheet(db, "atr2_2016_ignore", exportpath)
    cas.transfer_nonignore_records(db, "atr2_2016","expt_atr2")
    cas.rename_field(db,"expt_atr2","alloy name", "Alloy")
    dclean.standardize_alloy_names(db,"expt_atr2")
    cas.add_basic_field(db, "expt_atr2", "dataset", "ATR2")
    mcas.add_time_field(db, "expt_atr2")
    #cas.transfer_nonignore_records(db, "expt_ivar","expt_atr2")
    add_standard_fields(db, "expt_atr2")
    #Normalization
    add_normalized_fields(db, "expt_ivar", ["expt_ivar","expt_atr2","cd2_lwr"])
    #add_normalized_fields(db, "cd1_ivar", ["cd1_ivar","cd1_lwr"])
    #add_normalized_fields(db, "cd1_lwr", ["cd1_ivar","cd1_lwr"])
    add_normalized_fields(db, "cd2_ivar", ["cd2_ivar","cd2_lwr"])
    add_normalized_fields(db, "cd2_lwr", ["cd2_ivar","cd2_lwr"])
    add_normalized_fields(db, "expt_atr2", ["expt_ivar","expt_atr2","cd2_lwr"])
    #
    create_standard_conditions(db, "lwr_std_expt",3e10,290,3e6,5e9,["expt_ivar","expt_atr2","cd1_lwr"])
    create_standard_conditions(db, "atr2_std_expt",3.64e12,291,3e5,1.5e8,["expt_ivar","expt_atr2","cd1_lwr"])
    #create_standard_conditions(db, "lwr_std_cd1",3e10,290,3e6,5e9,["cd1_ivar","cd1_lwr"])
    create_standard_conditions(db, "lwr_std_cd2",3e10,290,3e6,5e9,["cd2_ivar","cd2_lwr"])
    #
    cas.export_spreadsheet(db, "expt_ivar", exportpath)
    #cas.export_spreadsheet(db, "cd1_ivar", exportpath)
    cas.export_spreadsheet(db, "cd2_ivar", exportpath)
    #cas.export_spreadsheet(db, "cd1_lwr", exportpath)
    cas.export_spreadsheet(db, "cd2_lwr", exportpath)
    cas.export_spreadsheet(db, "expt_atr2", exportpath)
    cas.export_spreadsheet(db, "lwr_std_expt", exportpath)
    #cas.export_spreadsheet(db, "lwr_std_cd1", exportpath)
    cas.export_spreadsheet(db, "lwr_std_cd2", exportpath)
    cas.export_spreadsheet(db, "atr2_std_expt", exportpath)
    #verify data
    #clist=["expt_ivar","cd1_ivar","cd2_ivar","cd1_lwr","cd2_lwr","expt_atr2"]
    clist=["expt_ivar","cd2_ivar","cd2_lwr","expt_atr2"]
    dver.make_per_alloy_plots(db, clist, "%s/verification_plots" % exportpath) 
    #Additional to-do
    ##
    return exportpath

if __name__ == "__main__":
    importpath = "../../../../data/DBTT_mongo/imports_201704"
    importpath = os.path.abspath(importpath)
    exportpath = main(importpath)
    print("Files in %s" % exportpath)
    sys.exit()
