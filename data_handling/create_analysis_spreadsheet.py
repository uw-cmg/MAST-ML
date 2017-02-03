#!/usr/bin/env python
##############
# Create analysis spreadsheet.
# Pull values from databases and add additional columns.
# TTM 2017-01-27
##############

import pymongo
import os
import sys
import subprocess 
import time
from pymongo import MongoClient
from bson.objectid import ObjectId
import data_transformations as dtf
import percent_converter

dbname="dbtt"

client = MongoClient('localhost', 27017)
db = client[dbname]

def get_nonignore_records(cname):
    results = db[cname].find({"ignore":{"$ne":1}})
    return results

def transfer_nonignore_records(fromcname, newcname, verbose=1):
    nonignore_records = get_nonignore_records(fromcname)
    nict = 0
    for nirecord in nonignore_records:
        db[newcname].insert(nirecord)
        nict = nict + 1
    if (verbose > 0):
        print("Transferred %i records." % nict)
    return

def match_and_add_records(newcname, records, matchlist=list(), matchas=list(), transferlist=list(), transferas=list(), matchmulti=False, verbose=1):
    """Match records to the collection and add certain fields
        Args:
            newcname <str>: Collection name to which to add records
            records <Cursor>: records to match
            matchlist <list of str>: fields on which to match values
            matchas <list of str>: corresponding match fields in the new collection
            transferlist <list of str>: field names to transfer
            transferas <list of str>: field names to rename upon transfer,
                    in the same order listed in transferlist
            matchmulti <bool>: True = yes, False = no (default)
    """
    uct=0
    if len(matchlist) == 0:
        raise ValueError("Matchlist cannot be empty. Must match on some fields.")
    for record in records:
        matchdict=dict()
        for midx in range(0, len(matchlist)):
            matchdict[matchas[midx]] = record[matchlist[midx]]
        setdict=dict()
        setdict["$set"]=dict()
        for tidx in range(0, len(transferlist)):
            setdict["$set"][transferas[tidx]] = record[transferlist[tidx]]
        updated = db[newcname].update(matchdict, setdict, multi=matchmulti)
        if updated['updatedExisting'] == True:
            uct = uct + 1
        if (verbose > 1):
            print("Updated: %s" % updated)
    if (verbose > 0):
        print("Total %i records updated." % uct)
    return

def list_all_fields(cname, verbose=0):
    """
        List all fields in a collection. Make more sophisticated later.
    """
    fieldlist=list()
    records = db[cname].find()
    for record in records:
        for key in record.keys():
            if not key in fieldlist:
                fieldlist.append(key)
    fieldlist.sort() #sort in place
    if verbose > 0:
        for field in fieldlist:
            print(field)
    return fieldlist

def export_spreadsheet(newcname="", prepath="", fieldlist=list()):
    if len(fieldlist) == 0:
        fieldlist=list_all_fields(newcname)
    fieldstr=""
    for field in fieldlist:
        fieldstr = fieldstr + field + ","
    
    outputpath = "%s_%s.csv" % (newcname, time.strftime("%Y%m%d_%H%M%S"))
    if not (prepath == ""):
        outputpath = os.path.join(prepath, outputpath)

    estr = "mongoexport"
    estr += " --db=%s" % dbname
    estr += " --collection=%s" % newcname
    estr += " --out=%s" % outputpath
    estr += " --type=csv"
    estr += ' --fields="%s"' % fieldstr
    eproc = subprocess.Popen(estr, shell=True,
                            stdout=subprocess.PIPE, 
                            stderr=subprocess.PIPE)
    print(eproc.communicate())
    eproc.wait()
    return



def main_ivar(newcname=""):
    transfer_nonignore_records("ucsbivarplus",newcname)
    print("Transferred experimental IVAR records.")
    cd_records = get_nonignore_records("cdivar2016") #change to 2017 later
    match_and_add_records(newcname, cd_records, 
        matchlist=["Alloy","flux_n_cm2_sec","fluence_n_cm2","temperature_C"],
        matchas=["Alloy","flux_n_cm2_sec","fluence_n_cm2","temperature_C"],
        transferlist= ["temperature_C","CD_delta_sigma_y_MPa"],
        transferas = ["CD_temperature_C","CD_delta_sigma_y_MPa"])
    print("Updated with CD IVAR temperature matches.")
    cd_records.rewind()
    match_and_add_records(newcname, cd_records, 
        matchlist=["Alloy","flux_n_cm2_sec","fluence_n_cm2","temperature_C"],
        matchas=["Alloy","flux_n_cm2_sec","fluence_n_cm2","original_reported_temperature_C"],
        transferlist= ["temperature_C","CD_delta_sigma_y_MPa"],
        transferas = ["CD_temperature_C","CD_delta_sigma_y_MPa"])
    print("Updated with CD IVAR temperature mismatches.")
    return

def main_lwr(newcname=""):
    transfer_nonignore_records("cdlwr2017", newcname)
    print("Transferred CD LWR records.")
    lwr_adjust_fields(newcname)
    print("Adusted some alloy names and units for flux and fluence.")
    ivar_records = get_nonignore_records("ucsbivarplus")
    match_and_add_records(newcname, ivar_records,
        matchlist=["Alloy"],
        matchas=["Alloy"],
        transferlist=["wt_percent_Cu","wt_percent_Ni","wt_percent_Mn",
                        "wt_percent_P","wt_percent_Si","wt_percent_C",
                        "product_id"],
        transferas=["wt_percent_Cu","wt_percent_Ni","wt_percent_Mn",
                        "wt_percent_P","wt_percent_Si","wt_percent_C",
                        "product_id"],
        matchmulti=True
                        )
    print("Updated with alloy weight percents.")
    return

def add_time_field(newcname, verbose=0):
    myfunc = getattr(dtf,"get_time_from_flux_and_fluence")
    records = db[newcname].find()
    for record in records:
        fieldval = myfunc(record["flux_n_cm2_sec"],record["fluence_n_cm2"])
        db[newcname].update(
            {'_id':record["_id"]},
            {"$set":{"time_sec":fieldval}}
            )
        if verbose > 0:
            print("Updated record %s with time %3f sec." % (record["_id"],fieldval))
    return

def add_converted_flux_and_fluence_field(newcname, verbose=0):
    records = db[newcname].find()
    for record in records:
        fluxval = record["flux_n_m2_sec"]
        fluenceval = record["fluence_n_m2"]
        newflux = fluxval / 1000.0
        newfluence = fluenceval / 1000.0
        db[newcname].update(
            {'_id':record["_id"]},
            {"$set":{"flux_n_cm2_sec":newflux, "fluence_n_cm2":newfluence}}
            )
        if verbose > 0:
            print("Updated record %s with flux %3.3f n/cm^2/sec and fluence %3.3f n/cm^2." % (record["_id"],newflux, newfluence))
    return

def modify_alloy_names(newcname, verbose=0):
    records = db[newcname].find()
    moddict=dict()
    moddict["WG"] = ["RR-WG"]
    moddict["WP"] = ["RR-WP"]
    moddict["Wv"] = ["RR-WV"]
    for record in records:
        alloy = record["Alloy"]
        if alloy in moddict.keys():
            replacename = moddict[alloy]
            db[newcname].update(
                {'_id':record["_id"]},
                {"$set":{"Alloy":replacename}}
                )
            if verbose > 0:
                print("Updated record %s with new alloy name %s" % replacename) 
    return

def add_atomic_percent_field(newcname, verbose=0):
    """Ignores percentages from Mo and Cr.
        Assumes balance is Fe.
    """
    elemlist=["Cu","Ni","Mn","P","Si","C"]
    records = db[newcname].find()
    for record in records:
        compdict=dict()
        for elem in elemlist:
            try:
                wt = float(record['wt_percent_%s' % elem])
            except (ValueError,TypeError):
                wt = 0.0
            compdict[elem] = wt
        compstr=""
        for elem in elemlist:
            compstr += "%s %s," % (elem, compdict[elem])
        compstr = compstr[:-1] #remove last comma
        outdict = percent_converter.main(compstr,"weight",0)
        db[newcname].update(
            {'_id':record["_id"]},
            {"$set":
                {"at_percent_Cu":outdict["Cu"]['perc_out'],
                "at_percent_Ni":outdict["Ni"]['perc_out'],
                "at_percent_Mn":outdict["Mn"]['perc_out'],
                "at_percent_P":outdict["P"]['perc_out'],
                "at_percent_Si":outdict["Si"]['perc_out'],
                "at_percent_C":outdict["C"]['perc_out']}
                }
            )
        if verbose > 0:
            print("Updated record %s with %s." % (record["_id"],outdict))
    for elem in elemlist:
        add_minmax_normalization_of_a_field(newcname, "at_percent_%s" % elem, 0.0, 1.717)
    return

def add_effective_fluence_field(newcname, verbose=0):
    """
        Calculated by fluence*(ref_flux/flux)^p where ref_flux = 3e10 n/cm^2/s and p=0.26
        IVAR has ref_flux of 3e11 n/cm^2/sec (Odette 2005)
        However, LWR conditions have lower flux, so use 3e10 n/cm^2/sec.
    """
    ref_flux=3.0e10 #n/cm^2/sec
    pvalue = 0.26
    myfunc = getattr(dtf,"get_effective_fluence")
    records = db[newcname].find()
    for record in records:
        fieldval = myfunc(flux=record["flux_n_cm2_sec"],
                            fluence=record["fluence_n_cm2"],
                            ref_flux=ref_flux,
                            pvalue=pvalue)
        db[newcname].update(
            {'_id':record["_id"]},
            {"$set":{"effective_fluence_n_cm2":fieldval}}
            )
        if verbose > 0:
            print("Updated record %s with effective fluence %3.2e n/cm2." % (record["_id"],fieldval))
    return

def add_generic_effective_fluence_field(newcname, ref_flux=3e10, pvalue=0.26, verbose=1):
    """
        Calculated by fluence*(ref_flux/flux)^p 
        IVAR has ref_flux of 3e11 n/cm^2/sec (Odette 2005)
        However, LWR conditions have lower flux, so use 3e10 n/cm^2/sec.
        Args:
            ref_flux <float>: reference flux in n/cm^2/sec
            pvalue <float>: p value
        Also adds a log10 field and a log10 min-max normalized field
    """
    myfunc = getattr(dtf,"get_effective_fluence")
    records = db[newcname].find()
    pvalstr = "%i" % (pvalue*100.0)
    newfield = "eff fl 100p=%s" % pvalstr
    for record in records:
        fieldval = myfunc(flux=record["flux_n_cm2_sec"],
                            fluence=record["fluence_n_cm2"],
                            ref_flux=ref_flux,
                            pvalue=pvalue)
        db[newcname].update(
            {'_id':record["_id"]},
            {"$set":{newfield:fieldval}}
            )
        if verbose > 0:
            print("Updated record %s with %s %3.2e n/cm2." % (record["_id"],newfield,fieldval))
    add_log10_of_a_field(newcname, newfield)
    add_minmax_normalization_of_a_field(newcname, "log(%s)" % newfield)
    return


def add_log10_of_a_field(newcname, origfield, verbose=0):
    """Add the base-10 logarithm of a field as a new field
    """
    newfield="log(%s)" % origfield
    myfunc = getattr(dtf,"get_log10")
    records = db[newcname].find()
    for record in records:
        fieldval = myfunc(record[origfield])
        db[newcname].update(
            {'_id':record["_id"]},
            {"$set":{newfield:fieldval}}
            )
        if verbose > 0:
            print("Updated record %s with %s %3.3f." % (record["_id"],newfield,fieldval))
    return

def add_product_type_columns(newcname, verbose=0):
    records = db[newcname].find()
    for record in records:
        pid = record['product_id'].strip().upper()
        plate=0
        weld=0
        SRM=0
        forging=0
        if pid == "P":
            plate=1
        elif pid == "W":
            weld=1
        elif pid == "SRM":
            SRM=1
        elif pid == "F":
            forging=1
        else:
            raise ValueError("Category for product id %s not found." % pid)
        db[newcname].update(
            {'_id':record["_id"]},
            {"$set":
                {"isPlate":plate,
                "isWeld":weld,
                "isForging":forging,
                "isSRM":SRM}
            }
            )
        if verbose > 0:
            print("Updated record %s with %s %3.3f." % (record["_id"],newfield,fieldval))

    return

def add_stddev_normalization_of_a_field(newcname, origfield, verbose=0):
    """Add the normalization of a field based on the mean and standard dev.
        Normalization is given as X_new = (X-X_mean)/(Xstd dev)
    """
    newfield="N(%s)" % origfield
    stddevagg = db[newcname].aggregate([
        {"$group":{"_id":None,"fieldstddev":{"$stdDevPop":"$%s" % origfield}}}
    ])
    for record in stddevagg:
        stddev = record['fieldstddev']
        print(record)
    avgagg = db[newcname].aggregate([
        {"$group":{"_id":None,"fieldavg":{"$avg":"$%s" % origfield}}}
    ])
    for record in avgagg:
        mean = record['fieldavg']
        print(record)
    print("Mean: %3.3f" % mean)
    print("StdDev: %3.3f" % stddev)
    records = db[newcname].find()
    for record in records:
        fieldval = ( record[origfield] - mean ) / (stddev)
        db[newcname].update(
            {'_id':record["_id"]},
            {"$set":{newfield:fieldval}}
            )
        if verbose > 0:
            print("Updated record %s with %s %3.3f." % (record["_id"],newfield,fieldval))
    return

def add_minmax_normalization_of_a_field(newcname, origfield, setmin=None, setmax=None, verbose=0):
    """Add the normalization of a field based on the min and max values
        Normalization is given as X_new = (X - X_min)/(X_max - X_min)
        For elemental compositions, max atomic percent is taken as 1.717 At%
            for Mn.
    """
    newfield="N(%s)" % origfield
    minagg = db[newcname].aggregate([
        {"$group":{"_id":None,"fieldminval":{"$min":"$%s" % origfield}}}
    ])
    for record in minagg:
        minval = record['fieldminval']
        print(record)
    maxagg = db[newcname].aggregate([
        {"$group":{"_id":None,"fieldmaxval":{"$max":"$%s" % origfield}}}
    ])
    for record in maxagg:
        maxval = record['fieldmaxval']
        print(record)
    print("Min: %3.3f" % minval)
    print("Max: %3.3f" % maxval)
    if not (setmin == None):
        minval = setmin
    if not (setmax == None):
        maxval = setmax
    records = db[newcname].find()
    for record in records:
        fieldval = ( record[origfield] - minval ) / (maxval - minval)
        db[newcname].update(
            {'_id':record["_id"]},
            {"$set":{newfield:fieldval}}
            )
        if verbose > 0:
            print("Updated record %s with %s %3.3f." % (record["_id"],newfield,fieldval))
    return

def add_eony_field(newcname, verbose=0):
    """Add the EONY 2013 model
    """
    records = db[newcname].find()
    for record in records:
        flux = record["flux_n_cm2_sec"]
        time = record["time_sec"]
        temp = record["temperature_C"]
        product_id = record["product_id"]
        wt_dict=dict()
        for element in ["Mn","Ni","Cu","P","Si","C"]:
            wt_dict[element] = record["wt_percent_%s" % element]
        ce_manuf= 0 #how to determine?
        eony_tts = dtf.get_eony_model_tts(flux, time, temp, product_id,
                                            wt_dict, ce_manuf, verbose=1)
        eony_delta_sigma_y = dtf.tts_to_delta_sigma_y(eony_tts, "F", product_id, verbose=1)
        db[newcname].update(
            {'_id':record["_id"]},
            {"$set":{"EONY_delta_sigma_y":eony_delta_sigma_y}}
            )
        if verbose > 0:
            print("Alloy %s flux %s fluence %s" % (record["Alloy"],flux,record["fluence_n_cm2"]))
            print("Updated record %s with %s %3.3f." % (record["_id"],"EONY_delta_sigma_y", eony_delta_sigma_y))
    return

def add_basic_field(newcname, fieldname,fieldval, verbose=0):
    records = db[newcname].find()
    for record in records:
        db[newcname].update(
            {'_id':record["_id"]},
            {"$set":{fieldname:fieldval}}
            )
        if verbose > 0:
            print("Updated record %s with value %s." % (record["_id"],fieldname, fieldval))
    return

def main_addfields(newcname=""):
    add_time_field(newcname)
    add_atomic_percent_field(newcname)
    add_effective_fluence_field(newcname)
    add_log10_of_a_field(newcname,"time_sec")
    add_log10_of_a_field(newcname,"fluence_n_cm2")
    add_log10_of_a_field(newcname,"flux_n_cm2_sec")
    add_log10_of_a_field(newcname,"effective_fluence_n_cm2")
    add_product_type_columns(newcname)
    add_minmax_normalization_of_a_field(newcname, "log(time_sec)")
    add_minmax_normalization_of_a_field(newcname, "log(fluence_n_cm2)")
    add_minmax_normalization_of_a_field(newcname, "log(flux_n_cm2_sec)")
    add_minmax_normalization_of_a_field(newcname, "time_sec")
    add_eony_field(newcname, 1)
    add_generic_effective_fluence_field(newcname, 3e10, 0.26)
    add_generic_effective_fluence_field(newcname, 3e10, 0.1)
    add_generic_effective_fluence_field(newcname, 3e10, 0.2)
    add_generic_effective_fluence_field(newcname, 3e10, 0.3)
    add_generic_effective_fluence_field(newcname, 3e10, 0.4)
    add_stddev_normalization_of_a_field(newcname, "delta_sigma_y_MPa")
    return

def lwr_addfields(newcname=""):
    add_atomic_percent_field(newcname)
    add_effective_fluence_field(newcname)
    add_log10_of_a_field(newcname,"time_sec")
    add_log10_of_a_field(newcname,"fluence_n_cm2")
    add_log10_of_a_field(newcname,"flux_n_cm2_sec")
    add_log10_of_a_field(newcname,"effective_fluence_n_cm2")
    add_product_type_columns(newcname)
    add_minmax_normalization_of_a_field(newcname, "log(time_sec)")
    add_minmax_normalization_of_a_field(newcname, "log(fluence_n_cm2)")
    add_minmax_normalization_of_a_field(newcname, "log(flux_n_cm2_sec)")
    add_minmax_normalization_of_a_field(newcname, "time_sec")
    add_eony_field(newcname, 1)
    add_generic_effective_fluence_field(newcname, 3e10, 0.26)
    add_generic_effective_fluence_field(newcname, 3e10, 0.1)
    add_generic_effective_fluence_field(newcname, 3e10, 0.2)
    add_generic_effective_fluence_field(newcname, 3e10, 0.3)
    add_generic_effective_fluence_field(newcname, 3e10, 0.4)
    #add_stddev_normalization_of_a_field(newcname, "delta_sigma_y_MPa")
    return

def lwr_adjust_fields(newcname=""):
    add_converted_flux_and_fluence_field(newcname)
    modify_alloy_names(newcname)
    add_basic_field(newcname, fieldname="temperature_C",fieldval=290)
    return

if __name__=="__main__":
    if len(sys.argv) > 2:
        ivarcname = sys.argv[1]
        lwrcname = sys.argv[2]
    else:
        ivarcname = "test_ivar_1"
        lwrcname = "test_lwr_1"
    #IVAR
    main_ivar(ivarcname)
    main_addfields(ivarcname)
    export_spreadsheet(ivarcname, "../../../data/DBTT_mongo/data_exports/")
    #LWR
    main_lwr(lwrcname)
    lwr_addfields(lwrcname)
    export_spreadsheet(lwrcname, "../../../data/DBTT_mongo/data_exports/")
