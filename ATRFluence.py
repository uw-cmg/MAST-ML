import matplotlib.pyplot as plt
import numpy as np
import operator


def execute(model, data, savepath, *args, **kwargs):

    '''
    Temp = 290 * np.ones((500, 1))
    Flux = 3e10 * np.ones((500, 1))
    aFluence = np.reshape(np.logspace(17, 21, 500), (500, 1))

    Temp = (Temp - np.min(np.asarray(data.get_data("Temp (C)")))) / (
        np.max(np.asarray(data.get_data("Temp (C)"))) - np.min(np.asarray(data.get_data("Temp (C)"))))
    Flux = (np.log10(Flux) - np.min(np.asarray(data.get_data("log(flux)")))) / (
        np.max(np.asarray(data.get_data("log(flux)"))) - np.min(np.asarray(data.get_data("log(flux)"))))
    Fluence = (np.log10(aFluence) - np.min(np.asarray(data.get_data("log(fluence)")))) / (
        np.max(np.asarray(data.get_data("log(fluence)"))) - np.min(np.asarray(data.get_data("log(fluence)"))))'''


    data.remove_all_filters()
    data.add_exclusive_filter("Data Set code", '==', 2)
    model.fit(data.get_x_data(), data.get_y_data())

    alloylist = [3, 5, 6, 7, 9, 11, 16, 17, 19, 20, 22, 31, 33, 34, 35, 36, 37, 38, 39, 45, 46, 47, 48, 49, 53]

    for alloy in alloylist:
        data.remove_all_filters()
        data.add_inclusive_filter("Alloy", '=', alloy)
        AlloyName = data.get_data("Alloy Name")[0][0]
        Alloy = np.reshape(np.asarray(data.get_x_data())[0, 0:6], (1, 6)) * np.ones((500, 6))



        fig, ax = plt.subplots()

        #Xdata = np.concatenate([Alloy, Fluence, Temp], 1)
        #ypredict = model.predict(Xdata)
        #ax.plot(np.log10(aFluence), ypredict, label='Model Prediction', c = 'black')



        data.add_exclusive_filter("Temp (C)", '<>', 290)
        data.add_exclusive_filter("Data Set code", '<>', 0)
        ax.scatter(data.get_data("log(fluence)"), data.get_y_data(), lw=0, color='#FF0034', label='IVAR', s = 40)
        ypredict = model.predict(data.get_x_data())
        xfl = data.get_data("log(fluence)")

        data.remove_all_filters()
        data.add_inclusive_filter("Alloy", '=', alloy)
        data.add_exclusive_filter("Temp (C)", '<>', 290)
        data.add_exclusive_filter("Data Set code", '<>', 1)
        if not len(data.get_x_data()) == 0:
            ax.scatter(data.get_data("log(fluence)"), data.get_y_data(), lw=0, color='#03FF00', label='ATR-1', s= 40)
            ypredict = np.append(ypredict, model.predict(data.get_x_data()))
            xfl = np.append(xfl, data.get_data("log(fluence)"))

        data.remove_all_filters()
        data.add_inclusive_filter("Alloy", '=', alloy)
        data.add_exclusive_filter("Data Set code", '<>', 2)
        ax.scatter(data.get_data("log(fluence)"), data.get_y_data(), lw=0, color='#49CCFF', label='ATR-2', s = 40)
        ypredict = np.append(ypredict, model.predict(data.get_x_data()))
        xfl = np.append(xfl, data.get_data("log(fluence)"))

        ax.scatter(xfl, ypredict, lw=0, color='black', label='Model Prediction', s = 40)
        xfl_sorted, ypredict_sorted = zip(*sorted(zip(xfl, ypredict),
                                                  key=operator.itemgetter(0), reverse=True))
        ax.plot(xfl_sorted, ypredict_sorted, color='black')

        ax.legend(loc='upper left')
        ax.set_title("{}({}) Fluence vs Delta Sigma, Temp = 290".format(alloy,AlloyName))
        ax.set_xlabel("log(Fluence(n/cm^2))")
        ax.set_ylabel("∆sigma (MPa)")
        fig.savefig(savepath.format(ax.get_title()), dpi=200, bbox_inches='tight')
        plt.close()
