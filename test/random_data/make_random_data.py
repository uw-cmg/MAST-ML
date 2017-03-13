#!/usr/bin/env python
#TTM make random data for testing
import numpy as np
import data_analysis.printout_tools as ptools
n_samples, n_features = 100, 5
rng = np.random.RandomState(0)
y = rng.randn(n_samples)
X = rng.randn(n_samples, n_features)

numpoints = len(y)
catlist=["A","B","C","D"]
catarr = list()
numarr = list()
for idx in range(0, numpoints):
    catnum = rng.randint(4)
    category = catlist[catnum]
    catarr.append(category)
    numarr.append(idx + 3)
catarr = np.array(catarr)
numarr = np.array(numarr)

array = np.array([numarr, catarr, X[:,0],X[:,1],X[:,2],X[:,3],X[:,4],y]).transpose()
headerstring = "numeric_label,category1,x1,x2,x3,x4,x5,y"

csvname="test.csv"

ptools.mixed_array_to_csv(csvname, headerstring, array)

