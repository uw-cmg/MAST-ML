import matplotlib
import matplotlib.pyplot as plt
import data_parser
import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error
from sklearn import cross_validation


def desimp(model=KernelRidge(alpha=.00139, coef0=1, degree=3, gamma=.518, kernel='rbf', kernel_params=None),
           datapath="../../DBTT_Data.csv", savepath='../../{}.png',
           X = ["N(Cu)", "N(Ni)", "N(Mn)", "N(P)", "N(Si)","N( C )", "N(log(fluence)", "N(log(flux)", "N(Temp)"], Y = "delta sigma"):
    data = data_parser.parse(datapath)
    data.set_x_features(X)
    data.set_y_feature(Y)

    overall_rms_list = []
    sd_list = []
    descriptorlist = ['Cu', 'Ni', 'Mn', 'P', 'Si', 'C', 'Fl', 'Fx', 'Temp']

    numFolds = 5
    numIter = 200
    model = model

    Xdata = data.get_x_data()
    Ydata = data.get_y_data().ravel()

    print("Testing descriptor importance using {}x {} - Fold CV".format(numIter, numFolds))
    print("")

    for x in range(len(data.get_x_data()[0])):
        RMS_List = []
        newX = np.delete(Xdata, x, 1)
        for n in range(numIter):
            kf = cross_validation.KFold(len(Xdata), n_folds=numFolds, shuffle=True)
            K_fold_rms_list = [];
            # split into testing and training sets
            for train_index, test_index in kf:
                X_train, X_test = newX[train_index], newX[test_index]
                Y_train, Y_test = Ydata[train_index], Ydata[test_index]
                # train on training sets
                model.fit(X_train, Y_train)
                YTP = model.predict(X_test)
                rms = np.sqrt(mean_squared_error(Y_test, YTP))
                K_fold_rms_list.append(rms)
            RMS_List.append(np.mean(K_fold_rms_list))
            # calculate rms

        maxRMS = np.amax(RMS_List)
        minRMS = np.amin(RMS_List)
        avgRMS = np.mean(RMS_List)
        medRMS = np.median(RMS_List)
        sd = np.sqrt(np.mean((RMS_List - np.mean(RMS_List)) ** 2))

        print("Removing {}:".format(descriptorlist[x]))
        print("The average RMSE was " + str(avgRMS))
        print("The median RMSE was " + str(medRMS))
        print("The max RMSE was " + str(maxRMS))
        print("The min RMSE was " + str(minRMS))
        print("The std deviation of the RMSE values was " + str(sd))
        print("")

        overall_rms_list.append(avgRMS)
        sd_list.append(sd)

    matplotlib.rcParams.update({'font.size': 15})
    fig, ax = plt.subplots()
    rects = ax.bar(np.arange(9), overall_rms_list, color='r', yerr=sd_list)
    ax.set_xlabel('Descriptor Removed')
    ax.set_ylabel('200x 5-fold RMSE')
    ax.set_title('Descriptor Importance')
    ax.set_xticks(np.arange(9) + .4)
    ax.set_xticklabels(descriptorlist)

    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2., 1.05 * height,
                '%.2f' % (height),
                ha='center', va='bottom')

    fig.savefig(savepath.format(plt.gca().get_title()), dpi=200, bbox_inches='tight')
    plt.show()
    plt.close()
