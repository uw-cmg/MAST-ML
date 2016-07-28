import matplotlib.pyplot as plt
import data_parser
import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error


def execute(model, data, savepath, lwr_data, *args):
    Xdata = np.asarray(data.get_x_data())
    Ydata = np.asarray(data.get_y_data()).ravel()

    model.fit(Xdata, Ydata)

    Temp = 290 * np.ones((500,1))
    Flux = 3e10 * np.ones((500,1))
    Time = np.reshape(np.linspace(3.154e+7/4, 3.154e+7 * 120, 500),(500,1))
    aFluence = (Flux * Time)
    #aFluence = np.reshape(np.logspace(16,21,100),(100,1))
    #normalize
    Temp = (Temp - np.min(np.asarray(data.get_data("Temp (C)"))))/(np.max(np.asarray(data.get_data("Temp (C)"))) - np.min(np.asarray(data.get_data("Temp (C)"))))
    Flux = (np.log10(Flux) - np.min(np.asarray(data.get_data("log(flux)"))))/(np.max(np.asarray(data.get_data("log(flux)"))) - np.min(np.asarray(data.get_data("log(flux)"))))
    Fluence  = (np.log10(aFluence) - np.min(np.asarray(data.get_data("log(fluence)"))))/(np.max(np.asarray(data.get_data("log(fluence)"))) - np.min(np.asarray(data.get_data("log(fluence)"))))

    for alloy in range(1, 60):
        data.remove_all_filters()
        data.add_inclusive_filter("Alloy", '=', alloy)

        if len(data.get_x_data()) == 0: continue  # if alloy doesn't exist(x data is empty), then continue

        Alloy = np.reshape(np.asarray(data.get_x_data())[0,0:6],(1,6)) * np.ones((500, 6))
        AlloyName = data.get_data("Alloy Name")[0][0]
        Xdata = np.concatenate([Alloy, Fluence, Flux, Temp], 1)

        ypredict = model.predict(Xdata)

        fig, ax = plt.subplots()
        ax.plot(np.log10(aFluence), ypredict, label='Model Prediction')

        if data.y_feature == "CD delta sigma" or data.y_feature == "EONY delta sigma":
            lwr_data.remove_all_filters()
            lwr_data.add_inclusive_filter("Alloy", '=', alloy)
            lwr_data.add_exclusive_filter("Temp (C)", '<>', 290)
            ax.scatter(np.log10(lwr_data.get_data("Fluence(n/cm^2)")), lwr_data.get_y_data(), lw = 0, label = lwr_data.y_feature)

        ax.legend(loc = 0)
        ax.set_title("{}({})".format(alloy,AlloyName))
        ax.set_xlabel("log(Fluence(n/cm^2))")
        ax.set_ylabel("predicted ∆sigma (MPa)")
        fig.savefig(savepath.format(ax.get_title()), dpi=200, bbox_inches='tight')
        plt.close()
