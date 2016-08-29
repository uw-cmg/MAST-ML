import numpy as np
import data_parser
import matplotlib
matplotlib.use('Agg')
import csv
import matplotlib.pyplot as plt
from sklearn import cross_validation
from sklearn.metrics import mean_squared_error

def lwr_extrap(model, data, lwr_data, scalings):
    lwr_data.remove_all_filters()

    x,y = np.asarray(data.get_x_data()),np.asarray(data.get_y_data()).ravel()
    lwr_x,lwr_y = np.asarray(lwr_data.get_x_data()), np.asarray(lwr_data.get_y_data()).ravel()

    for i in range(len(scalings)):
        x[:,i] = x[:,i]*scalings[i]
        lwr_x[:,i] = lwr_x[:,i]*scalings[i]

    model.fit(x,y)
    overall_rmse = np.sqrt(mean_squared_error(model.predict(lwr_x), lwr_y))

    lwr_data.add_exclusive_filter("Time(Years)", '<', 60)
    lwr_data.add_exclusive_filter("Time(Years)", '>', 100)
    lwr_x, lwr_y = np.asarray(lwr_data.get_x_data()), np.asarray(lwr_data.get_y_data()).ravel()
    for i in range(len(scalings)):
        lwr_x[:, i] = lwr_x[:, i] * scalings[i]

    rmse_60100 =  np.sqrt(mean_squared_error(model.predict(lwr_x), lwr_y))
    return (overall_rmse,rmse_60100)



def kfold_cv(model, X, Y, num_folds=5, num_runs=200):
    Xdata = X
    Ydata = Y

    RMS_List = []
    for n in range(num_runs):
        kf = cross_validation.KFold(len(Xdata), n_folds=num_folds, shuffle=True)
        K_fold_rms_list = []
        # split into testing and training sets
        for train_index, test_index in kf:
            X_train, X_test = Xdata[train_index], Xdata[test_index]
            Y_train, Y_test = Ydata[train_index], Ydata[test_index]
            # train on training sets
            model.fit(X_train, Y_train)
            Y_test_Pred = model.predict(X_test)
            rms = np.sqrt(mean_squared_error(Y_test, Y_test_Pred))
            K_fold_rms_list.append(rms)

        RMS_List.append(np.mean(K_fold_rms_list))

    return {'rms':np.mean(RMS_List), 'std':np.std(RMS_List)}


def alloy_cv(model, X, Y, AlloyList):
    rms_list = []
    for alloy in range(1, 60):
        train_index = []
        test_index = []
        for x in range(len(AlloyList)):
            if AlloyList[x] == alloy:
                test_index.append(x)
            else:
                train_index.append(x)
        if len(train_index) == 0 or len(test_index) == 0: continue  # if alloy doesn't exist(x data is empty), then continue
        # fit model to all alloys except the one to be removed
        model.fit(X[train_index], Y[train_index])

        Ypredict = model.predict(X[test_index])

        rms = np.sqrt(mean_squared_error(Ypredict, Y[test_index]))
        rms_list.append(rms)

    return {'rms': np.mean(rms_list), 'std': np.std(rms_list)}


def execute (model, data, savepath, lwr_data, *args, **kwargs):


    Alloys = data.get_data("Alloy")
    results = []
    for i in np.arange(.5, 4.5, .5):
        for j in np.arange(.5, 4.5, .5):
            for k in np.arange(.5, 4.5, .5):
                for l in np.arange(.5, 4.5, .5):
                    for m in np.arange(.5, 4.5, .5):
                        scalings = [i,j,k,l,m]
                        rms = lwr_extrap(model,data,lwr_data,scalings)
                        results.append([i,j,k,l,m,rms[0],rms[1]])

    with open(savepath.replace(".png", "").format("scalings.csv"), 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        x = ["Cu", "Ni", "Mn", "P", "Fl", "Overall RMSE", "60-100 RMSE"]
        writer.writerow(x)
        for i in results:
            writer.writerow(i)

            '''
            newX[:, i] = newX[:, i] * j
            #kfold = kfold_cv(model, X = newX, Y = Ydata, num_folds = 5, num_runs = 200)
            #alloy = alloy_cv(model, newX, Ydata, Alloys)
            #ax.errorbar(j,kfold['rms'],yerr = kfold['std'], c = 'red', label = '5-fold CV', fmt='o')
            #ax.errorbar(j, alloy['rms'], yerr = alloy['std'], c = 'blue', label = 'Alloy CV', fmt='o')
            ax.errorbar(j, kfold['rms'] + alloy['rms'], yerr = kfold['std'] + alloy['std'], c = 'm', fmt='o')
            print(i, j, kfold['rms'], alloy['rms'], kfold['rms'] + alloy['rms'])
        ax.set_xlabel("Scale Factor")
        ax.set_ylabel("RMSE")
        ax.set_title(X[i])
        fig.savefig(savepath.format(plt.gca().get_title()), dpi=200, bbox_inches='tight')
        fig.clf()
        plt.close()'''
