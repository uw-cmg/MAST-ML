import matplotlib.pyplot as plt
import data_parser
import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error

'''
Plot CD predictions of ∆sigma for data and lwr_data as well as model's output
model is trained to data, which is in the domain of IVAR, IVAR+/ATR1 (assorted fluxes and fluences)
lwr_data is data in the domain of LWR conditions (low flux, high fluence)
'''

def execute(model, data, savepath, lwr_data, *args, **kwargs):
    Xdata = np.asarray(data.get_x_data())
    Ydata = np.asarray(data.get_y_data()).ravel()

    model.fit(Xdata, Ydata)

    Flux = 3e10 * np.ones((500,1))
    Time = np.reshape(np.linspace(3.154e+7/4, 3.154e+7 * 150, 500),(500,1))
    aFluence = (Flux * Time)
    aEffectiveFluence = aFluence
    combinedfluence = data.get_data("log(fluence)") + lwr_data.get_data("log(fluence)")
    combinedflux = data.get_data("log(flux)") + lwr_data.get_data("log(flux)")
    combinedeffectivefluence = data.get_data("log(eff fluence)") + lwr_data.get_data("log(eff fluence)")
    #normalize
    Flux = (np.log10(Flux) - np.min(combinedflux))/(np.max(combinedflux) - np.min(combinedflux))
    Fluence  = (np.log10(aFluence) - np.min(combinedfluence))/(np.max(np.asarray(combinedfluence)) - np.min(combinedfluence))
    EffectiveFluence = (np.log10(aEffectiveFluence) - np.min(combinedeffectivefluence))/(np.max(np.asarray(combinedeffectivefluence)) - np.min(combinedeffectivefluence))

    for alloy in range(1, 60):
        data.remove_all_filters()
        data.add_inclusive_filter("Alloy", '=', alloy)
        if len(data.get_x_data()) == 0: continue  # if alloy doesn't exist(x data is empty), then continue
        AlloyName = data.get_data("Alloy Name")[0][0]

        fig, ax = plt.subplots()


        lwr_data.remove_all_filters()
        lwr_data.add_inclusive_filter("Alloy", '=', alloy)
        lwr_data.add_exclusive_filter("Temp (C)", '<>', 290)





        if not data.y_feature == "delta sigma":

            ax.plot(np.log10(lwr_data.get_data("fluence n/cm2")), model.predict(lwr_data.get_x_data()), lw=3,
                    color='#ffc04d', label="CD LWR prediction")
            ax.scatter(np.log10(lwr_data.get_data("fluence n/cm2")), lwr_data.get_y_data(), lw=0,
                       label="CD LWR data", color = '#7ec0ee')
            ax.scatter(np.log10(data.get_data("fluence n/cm2")), data.get_y_data(), lw=0, label='CD IVAR data',
                       color='black')

        else:
            ax.scatter(np.log10(data.get_data("fluence n/cm2")), data.get_y_data(), lw=0, label='Expt IVAR data',
                       color='green')
            if len(lwr_data.get_x_data()) == 0:
                Alloy = np.reshape(np.asarray(data.get_x_data())[0, 0:6], (1, 6)) * np.ones((500, 6))
                Xdata = np.concatenate([Alloy, EffectiveFluence], 1)
                ypredict = model.predict(Xdata)
                ax.plot(np.log10(aEffectiveFluence), ypredict, lw = 3, label='Model Prediction', color = 'orange')
            else:
                ax.plot(np.log10(lwr_data.get_data("fluence n/cm2")), model.predict(lwr_data.get_x_data()), lw=3,
                        color='orange', label="Expt LWR prediction")



        ax.legend(loc = 0)
        ax.set_title("{}({})".format(alloy,AlloyName))
        ax.set_xlabel("log(Fluence(n/cm^2))")
        ax.set_ylabel("∆sigma (MPa)")
        fig.savefig(savepath.format(ax.get_title()), dpi=200, bbox_inches='tight')
        plt.close()