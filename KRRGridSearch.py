import numpy as np
import csv
import data_parser
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn import cross_validation
from sklearn.metrics import mean_squared_error
from sklearn.kernel_ridge import KernelRidge
from matplotlib import cm as cm

def lwr_extrap(model, data, lwr_data):
    if (data.y_feature == "delta sigma"):
        print("Must set Y to be CD Delta Sigma or EONY Delta Sigma for Extrapolating to LWR")
        return
    scalings = [4,2,2,1,4]
    trainX = np.asarray(data.get_x_data())
    trainY = np.asarray(data.get_y_data()).ravel()
    lwr_data.remove_all_filters()
    lwr_data.add_exclusive_filter("Time(Years)", '<', 60)
    lwr_data.add_exclusive_filter("Time(Years)", '>', 100)
    testX = np.asarray(lwr_data.get_x_data())
    testY =  np.asarray(lwr_data.get_y_data()).ravel()
    #for x in range(len(scalings)):
    #    trainX[:,x] = trainX[:,x]*scalings[x]
    #    testX[:, x] = testX[:, x]*scalings[x]
    model.fit(trainX, trainY)


    return np.sqrt(mean_squared_error(model.predict(testX), testY))

def atr2_extrap(model, data):
    data.remove_all_filters()
    data.add_exclusive_filter("Data Set code", '=', 2)

    x = np.asarray(data.get_x_data())
    y = np.asarray(data.get_y_data()).ravel()
    model.fit(x, y)

    data.remove_all_filters()
    data.add_inclusive_filter("Data Set code", '=', 2)

    x = np.asarray(data.get_x_data())
    y = np.asarray(data.get_y_data()).ravel()

    y_predict = model.predict(x)

    return np.sqrt(mean_squared_error(y, y_predict))

def kfold_cv(model, data, num_folds=5, num_runs=200):

    Xdata = np.asarray(data.get_x_data())
    Ydata = np.asarray(data.get_y_data()).ravel()

    RMS_List = []
    for n in range(num_runs):
        kf = cross_validation.KFold(len(Xdata), n_folds=num_folds, shuffle=True)
        K_fold_rms_list = []
        # split into testing and training sets
        for train_index, test_index in kf:
            X_train, X_test = Xdata[train_index], Xdata[test_index]
            Y_train, Y_test = Ydata[train_index], Ydata[test_index]
            # train on training sets
            model = model
            model.fit(X_train, Y_train)
            Y_test_Pred = model.predict(X_test)
            rms = np.sqrt(mean_squared_error(Y_test, Y_test_Pred))
            K_fold_rms_list.append(rms)

        RMS_List.append(np.mean(K_fold_rms_list))

    avgRMS = np.mean(RMS_List)
    #print (avgRMS)
    return avgRMS

def leaveout_cv(model, data, testing_size = .95, num_runs=200):

    Xdata = data.get_x_data()
    Ydata = np.asarray(data.get_y_data()).ravel()

    RMS_List = []
    for n in range(num_runs):
        # split into testing and training sets
        X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(Xdata, Ydata, test_size= testing_size)
        model.fit(X_train, Y_train)
        Y_test_Pred = model.predict(X_test)
        rms = np.sqrt(mean_squared_error(Y_test, Y_test_Pred))
        RMS_List.append(rms)

    avgRMS = np.mean(RMS_List)

    #print (avgRMS)
    return avgRMS

def alloy_cv(model, data):

    rms_list = []
    for alloy in range(1, 60):
        model = model  # creates a new model

        # fit model to all alloys except the one to be removed
        data.remove_all_filters()
        data.add_exclusive_filter("Alloy", '=', alloy)
        model.fit(data.get_x_data(), data.get_y_data().ravel())

        # predict removed alloy
        data.remove_all_filters()
        data.add_inclusive_filter("Alloy", '=', alloy)
        if len(data.get_x_data()) == 0: continue  # if alloy doesn't exist(x data is empty), then continue
        Ypredict = model.predict(data.get_x_data())

        rms = np.sqrt(mean_squared_error(Ypredict, data.get_y_data().ravel()))
        rms_list.append(rms)

    return np.mean(rms_list)

#todo kwarg the choice of CV type to pass in config
def execute(model, data, savepath,  lwr_data, *args, **kwargs):


    parameters = {'alpha' : np.logspace(-6,0,20), 'gamma': np.logspace(-1.5,1.5,20)}
    grid_scores = []

    for a in parameters['alpha']:
        for y in parameters['gamma']:
            model = KernelRidge(alpha= a, gamma= y, kernel='rbf')
            rms = leaveout_cv(model, data, num_runs = 200)
            #rms = kfold_cv(model, data, num_folds = 5, num_runs = 50)
            #rms = alloy_cv(model, data)
            #rms = atr2_extrap(model, data)
            #rms = lwr_extrap(model, data, lwr_data)
            grid_scores.append((a, y, rms))

    grid_scores = np.asarray(grid_scores)

    with open(savepath.replace(".png","").format("grid_scores.csv"),'w') as f:
        writer = csv.writer(f, lineterminator = '\n')
        x = ["alpha", "gamma", "rms"]
        writer.writerow(x)
        for i in grid_scores:
            writer.writerow(i)

    #Heatmap of RMS scores vs the alpha and gamma
    plt.figure(1)
    plt.hexbin(np.log10(grid_scores[:,0]), np.log10(grid_scores[:,1]), C = grid_scores[:,2], gridsize=15, cmap=cm.plasma, bins=None, vmax = 60)
    plt.xlabel('log alpha')
    plt.ylabel('log gamma')
    cb = plt.colorbar()
    cb.set_label('rms')
    plt.savefig(savepath.format("alphagammahex"), dpi=200, bbox_inches='tight')
    plt.close()

