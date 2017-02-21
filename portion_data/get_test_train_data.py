#!/usr/bin/env python
###################
# Get test and train data
# Tam Mayeshiba 2017-02-21
###################
import numpy as np
import os
import sys
import traceback
import time
from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneGroupOut

def get_kfold_indices(datalen, num_folds=5, verbose=1):    
    kf = KFold(n_splits=num_folds)
    dummy_arr = np.arange(0, datalen)
    indices = dict()
    nfold = 0
    for train_index, test_index in kf.split(dummy_arr):
        nfold = nfold + 1
        indices[nfold] = dict()
        indices[nfold]["train_index"] = train_index
        indices[nfold]["test_index"] = test_index
    if verbose > 0:
        ikeys = list(indices.keys())
        ikeys.sort()
        for ikey in ikeys:
            print("Fold:", ikey)
            for label in ["train_index","test_index"]:
                print("%s:" % label, indices[ikey][label])
    return indices

def get_logo_indices(grouparr, verbose=1):
    logo = LeaveOneGroupOut()
    dummy_arr = np.arange(0, len(grouparr))
    indices = dict()
    nfold = 0
    for train_index, test_index in logo.split(dummy_arr, groups=grouparr):
        nfold = nfold + 1
        indices[nfold] = dict()
        indices[nfold]["train_index"] = train_index
        indices[nfold]["test_index"] = test_index
    if verbose > 0:
        ikeys = list(indices.keys())
        ikeys.sort()
        for ikey in ikeys:
            print("Group:", ikey)
            for label in ["train_index","test_index"]:
                print("%s:" % label, indices[ikey][label])
    return indices

def get_field_grouping_array(data, grouping_field="", field_type=0):
    """
        Args:
            data <data_parser type dataset>
            grouping_field <str>: grouping field name, like "alloy_number"
            field_type <int>: 0 - integer, string, or other type that
                                    immediately separates into groups
                              no other values are supported yet, but
                                could imagine splitting into bins
    """
    grouparr = None
    fieldarr = np.asarray(data.get_data(grouping_field)).ravel()
    dlen = len(fieldarr)
    if field_type == 0:
        grouparr = fieldarr
    else:
        raise ValueError("Field type %i not supported" % field_type)
    return grouparr

def get_field_grouping_indices(data, grouping_field="", field_type=0):
    """
        Args:
            data <data_parser type dataset>
            grouping_field <str>: grouping field name, like "alloy_number"
            field_type <int>: 0 - see get_field_grouping_array
    """
    grouparr = get_field_grouping_array(data, grouping_field, field_type)
    indices = get_logo_indices(grouparr)
    return indices

if __name__=="__main__":
    #kfold(X, y, 3)
    get_kfold_indices(12,3,1)
    get_logo_indices([1,1,1,2,1,2,3,10,10])
    import data_parser
    data=data_parser.parse("testdata.csv")
    get_field_grouping_indices(data, "name")
    get_field_grouping_indices(data, "number")
    get_field_grouping_indices(data, "dataset")



