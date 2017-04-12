#!/usr/bin/env python
#TTM make random data for testing
import numpy as np
import data_analysis.printout_tools as ptools
n_samples, n_features = 100, 5
rng = np.random.RandomState()
y = rng.randn(n_samples)
X = rng.randn(n_samples, n_features)

numpoints = int(n_samples/4)
numcats = 4
catlist=["A","B","C","D"]
num_id_arr = list()
num_cat_arr = list()
str_cat_arr = list()
time_arr = list()
sine_feature_arr = list()
linear_feature_arr = list()
sine_feature_error_arr = list()
linear_feature_error_arr = list()
y_feature_arr = list()
nidx = 0
for idx in range(0, numpoints):
    time = idx * np.pi/6.0
    for cidx in range(0, 4):
        catnum = rng.randint(4)
        category = catlist[catnum]
        str_cat_arr.append(category)
        num_cat_arr.append(catnum)
        num_id_arr.append(nidx)
        time_arr.append(time)
        sine_feature_arr.append(np.sin(time) + X[nidx,0]) #add noise
        sine_feature_error_arr.append(X[nidx,3]/X[nidx,4]/100.0)
        linear_feature_arr.append(100*time + 30.0 + X[nidx,1]) #add noise
        linear_feature_error_arr.append(X[nidx,2]/X[nidx,3]/100.0)
        y_feature_arr.append(np.sin(time) + y[nidx])
        nidx = nidx + 1

num_id_arr = np.array(num_id_arr)
num_cat_arr = np.array(num_cat_arr)
str_cat_arr = np.array(str_cat_arr)
time_arr = np.array(time_arr)
sine_feature_arr = np.array(sine_feature_arr)
sine_feature_error_arr = np.array(sine_feature_error_arr)
linear_feature_arr = np.array(linear_feature_arr)
linear_feature_error_arr = np.array(linear_feature_error_arr)
y_feature_arr = np.array(y_feature_arr)

array = np.array([num_id_arr, 
            num_cat_arr, 
            str_cat_arr, 
            time_arr,
            sine_feature_arr,sine_feature_error_arr,
            linear_feature_arr,linear_feature_error_arr,
            y_feature_arr]).transpose()
headerstring = "num_id,num_cat,str_cat,time,sine_feature,sine_error,linear_feature,linear_error,y_feature"

csvname="test.csv"

ptools.mixed_array_to_csv(csvname, headerstring, array)

