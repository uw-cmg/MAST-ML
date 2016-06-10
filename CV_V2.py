# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 15:25:14 2016

@author: asus
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 15:07:36 2016

@author: asus
"""

import numpy as np
from sklearn import cross_validation
from sklearn.metrics import mean_squared_error
import data_parser


def cv(datapath,model,fold,run):
    #get data
    data = data_parser.parse(datapath)
    data.set_x_features(["N(Cu)", "N(Ni)", "N(Mn)", "N(P)", "N(Si)","N( C )", "N(log(fluence)", "N(log(flux)", "N(Temp)"])
    data.set_y_feature("delta sigma")
    Ydata = data.get_y_data()
    Xdata = data.get_x_data()
    Y_new = Ydata - np.mean(Ydata)

    RMS_List = [] 
    for n in range(run):
        kf = cross_validation.KFold(len(Xdata),n_folds = fold, shuffle = True)
        K_fold_rms_list = []
        Overall_Y_Pred = np.zeros(len(Xdata))
        #split into testing and training sets
        for train_index, test_index in kf:
            X_train, X_test = Xdata[train_index], Xdata[test_index]
            Y_train, Y_test = Y_new[train_index], Y_new[test_index]
            #train on training sets
            model.fit(X_train, Y_train)
            Y_test_Pred= model.predict(X_test) 
            rms = np.sqrt(mean_squared_error(Y_test,Y_test_Pred))
            K_fold_rms_list.append(rms)
            Overall_Y_Pred[test_index]= Y_test_Pred 
            
        RMS_List.append(np.mean(K_fold_rms_list))        
    avgRMS = np.mean(RMS_List)
    print ("You did a "+str(fold)+" fold cross validation using "+ datapath)
    print ("mean_RMSE: " + str(avgRMS))
    

########################################################################
#things need to be changed before you run the codes
#datapath = "DBTT_Data9.csv"
#from sklearn import tree
#model = tree.DecisionTreeRegressor(max_depth=14, min_samples_split=3, min_samples_leaf=1)
#fold = 2
#run = 200
#########################################################################
cv(datapath,model,fold,run)
