import numpy as np
import matplotlib.pyplot as plt
from sklearn import cross_validation
from sklearn.metrics import mean_squared_error
import data_parser

def cv(datapath,model,fold,run):      
    # imports csv file
    '''f = open(datapath)
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
    Xdata = np.concatenate((XCu, XNi, XMn, XP, XSi, XC, XFl, XFx, XT), axis=1)'''

    
    #get data
    data = data_parser.parse(datapath)
    data.set_x_features(["N(Cu)", "N(Ni)", "N(Mn)", "N(P)", "N(Si)","N( C )", "N(log(fluence)", "N(log(flux)", "N(Temp)"])
    data.set_y_feature("delta sigma")
    Ydata = data.get_y_data()
    Xdata = data.get_x_data()
    
    Y_new = Ydata - np.mean(Ydata)
    Y_predicted_best = []
    Y_predicted_worst = []
    
    maxRMS = 10
    minRMS = 40
    
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
            MODEL = model
            MODEL.fit(X_train, Y_train)
            Y_test_Pred= MODEL.predict(X_test)
            rms = np.sqrt(mean_squared_error(Y_test,Y_test_Pred))
            K_fold_rms_list.append(rms)
            Overall_Y_Pred[test_index]= Y_test_Pred 
            
        RMS_List.append(np.mean(K_fold_rms_list))
        
        if np.mean(K_fold_rms_list) > maxRMS:
            maxRMS = np.mean(K_fold_rms_list)
            Y_predicted_worst = Overall_Y_Pred
            
        if np.mean(K_fold_rms_list) < minRMS:  
            minRMS = np.mean(K_fold_rms_list)
            Y_predicted_best = Overall_Y_Pred
    
    
    avgRMS = np.mean(RMS_List)
    medRMS = np.median(RMS_List)
    sd = np.sqrt(np.mean((RMS_List - np.mean(RMS_List)) ** 2))
    
    print ("Using %i - Fold CV and decision tree regression")
    print ("The average RMSE was " + str(avgRMS))
    print ("The median RMSE was " + str(medRMS))
    print ("The max RMSE was " + str(maxRMS))
    print ("The min RMSE was " + str(minRMS))
    print ("The std deviation of the RMSE values was " + str(sd))
    
    font = {'family' : 'normal',
            'weight' : 'bold',
            'size'   : 13}
    plt.rc('font', **font)
    
    plt.figure(1)
    plt.scatter(Y_new,Y_predicted_best, c='black')
    plt.plot((-50,700),(-50,700),color = 'blue', linewidth = 3)
    plt.title('Best Fit')
    plt.figtext(.15,.83,'Min RMSE: ' + str(minRMS) , fontsize = 15)
    plt.xlabel('Measured (Mpa)')
    plt.ylabel('Predicted (Mpa)')
    plt.savefig('cv_best_fit.png',dpi=200, bbox_inches= 'tight') 
    
    plt.figure(2)
    plt.scatter(Y_new,Y_predicted_worst,c='black') 
    plt.plot((-50,700),(-50,700),color = 'blue', linewidth = 3) 
    plt.title('Worst Fit')
    plt.figtext(.15,.83,'Max RMSE: ' + str(maxRMS) , fontsize = 15)
    plt.xlabel('Measured (Mpa)')
    plt.ylabel('Predicted (Mpa)')   
    plt.savefig('cv_worst_fit.png',dpi=200, bbox_inches= 'tight')  
  
########################################################################
#things need to be changed before you run the codes
datapath = "DBTT_Data9.csv"

#from sklearn import tree
#model = tree.DecisionTreeRegressor(max_depth=14, min_samples_split=3, min_samples_leaf=1)
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=100,
                                            max_features='auto',
                                            max_depth=13,
                                            min_samples_split=3,
                                            min_samples_leaf=1,
                                            min_weight_fraction_leaf=0,
                                            max_leaf_nodes=None,
                                            n_jobs=1)
#from sklearn.kernel_ridge import KernelRidge
#model = KernelRidge(alpha= 0.00139, gamma=0.518, kernel='rbf')
fold = 5
run = 200
#########################################################################
cv(datapath,model,fold,run)
