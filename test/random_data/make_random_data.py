#!/usr/bin/env python
#TTM make random data for testing
import numpy as np
import data_analysis.printout_tools as ptools
n_samples, n_features = 100, 5
rng = np.random.RandomState(0)
y = rng.randn(n_samples)
X = rng.randn(n_samples, n_features)

numpoints = len(y)
catlist=["A","B","C","D","E","F","G","H"]
num_id_arr = list()
num_cat_arr = list()
str_cat_arr = list()
for idx in range(0, numpoints):
    catnum = rng.randint(8)
    category = catlist[catnum]
    str_cat_arr.append(category)
    num_cat_arr.append(catnum)
    num_id_arr.append(idx)
num_id_arr = np.array(num_id_arr)
num_cat_arr = np.array(num_cat_arr)
str_cat_arr = np.array(str_cat_arr)
array = np.array([num_id_arr, 
            num_cat_arr, 
            str_cat_arr, 
            X[:,0],X[:,1],X[:,2],X[:,3],X[:,4],y]).transpose()
headerstring = "num_id,num_cat,str_cat,x1,x2,x3,x4,x5,y"

csvname="test.csv"

ptools.mixed_array_to_csv(csvname, headerstring, array)

