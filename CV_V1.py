#instruction: at the bottom, modify file,model and fold and click run to get result of cross validation

import numpy as np
from sklearn import cross_validation
from sklearn.metrics import mean_squared_error

def cv(file,model,fold):    
    # imports csv file
    f = open("DBTT_Data9.csv")
    title = f.readline().split(',')
    data =  np.loadtxt(f, delimiter = ',')
    
    dataTemp = data[:, np.newaxis]
    # converts each column in the excell file into an array
    XCu = (dataTemp[:, :, title.index("N(Cu)")])
    XNi = (dataTemp[:, :, title.index("N(Ni)")])
    XMn = (dataTemp[:, :, title.index("N(Mn)")])
    XP  = (dataTemp[:, :, title.index("N(P)")])
    XSi = (dataTemp[:, :, title.index("N(Si)")])
    XC  = (dataTemp[:, :, title.index("N( C )")])
    XFl = (dataTemp[:, :, title.index("N(log(fluence)")])
    XFx = (dataTemp[:, :, title.index("N(log(flux)")])
    XT  = (dataTemp[:, :, title.index("N(Temp)")])
    
    Ydata = (dataTemp[:, :, title.index("delta sigma")])
    Xdata = np.concatenate((XCu, XNi, XMn, XP, XSi, XC, XFl, XFx, XT), axis=1)
    
    Y_new = Ydata - np.mean(Ydata)
    
    RMS_List = []
    num_folds = 5
    for n in range(200):
        kf = cross_validation.KFold(len(Xdata),n_folds = num_folds, shuffle = True)
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
    return avgRMS

#########################################################################
#change the file if needed
file = "DBTT_Data9.csv" 

#line 62-63: change the model and its parameters if needed
from sklearn import tree 
model = tree.DecisionTreeRegressor(max_depth=14, min_samples_split=3, min_samples_leaf=1)

#change fold number if needed
fold = 5
#########################################################################

RMSE= cv(file,model,fold)
print()
print ("You did a "+str(fold)+" fold cross validation using "+ file)
print()
print ("mean_RMSE: " + str(RMSE))