import numpy as np
import data_parser
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn import cross_validation
from sklearn.metrics import mean_squared_error


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


def execute (model, data, savepath, *args):

    Ydata = data.get_y_data().ravel()
    Xdata = data.get_x_data()
    Alloys = data.get_data("Alloy")

    for i in range(len(Xdata[0])):
        rms_list = []
        fig, ax = plt.subplots()
        for j in np.arange(0, 5.5, .5):
            newX = np.copy(Xdata)
            newX[:, i] = newX[:, i] * j
            kfold = kfold_cv(model, X = newX, Y = Ydata, num_folds = 5, num_runs = 200)
            alloy = alloy_cv(model, newX, Ydata, Alloys)
            #ax.errorbar(j,kfold['rms'],yerr = kfold['std'], c = 'red', label = '5-fold CV', fmt='o')
            #ax.errorbar(j, alloy['rms'], yerr = alloy['std'], c = 'blue', label = 'Alloy CV', fmt='o')
            ax.errorbar(j, kfold['rms'] + alloy['rms'], yerr = kfold['std'] + alloy['std'], c = 'm', fmt='o')
            print(i, j, kfold['rms'], alloy['rms'], kfold['rms'] + alloy['rms'])
        ax.set_xlabel("Scale Factor")
        ax.set_ylabel("RMSE")
        ax.set_title(X[i])
        fig.savefig(savepath.format(plt.gca().get_title()), dpi=200, bbox_inches='tight')
        fig.clf()
        plt.close()

from sklearn.kernel_ridge import KernelRidge
model = KernelRidge(alpha = .00518, gamma = .518, kernel = 'laplacian')
X=["N(Cu)", "N(Ni)", "N(Mn)", "N(P)","N(Si)", "N( C )", "N(log(fluence)", "N(log(flux)", "N(Temp)"]
Y="delta sigma"
datapath="../../DBTT_Data.csv"
savepath='../../bardeengraphs/{}.png'
data = data_parser.parse(datapath)
data.set_x_features(X)
data.set_y_feature(Y)

execute(model, data, savepath)