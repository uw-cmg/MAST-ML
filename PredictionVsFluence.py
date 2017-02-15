import matplotlib.pyplot as plt
import matplotlib
import data_parser
import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error
import data_analysis.printout_tools as ptools

'''
Plot CD predictions of ∆sigma for data and lwr_data as well as model's output
model is trained to data, which is in the domain of IVAR, IVAR+/ATR1 (assorted fluxes and fluences)
lwr_data is data in the domain of LWR conditions (low flux, high fluence)
'''

def execute(model, data, savepath, lwr_data, *args, **kwargs):
    Xdata = np.asarray(data.get_x_data())
    Ydata = np.asarray(data.get_y_data()).ravel()

    model.fit(Xdata, Ydata)

    fluence_str = "log(fluence_n_cm2)"
    flux_str = "log(flux_n_cm2_sec)"
    eff_str = "log(eff fl 100p=26)"
    temp_str = "temperature_C"

    #TTM comment out block until determine why renormalize columns
    #Flux = 3e10 * np.ones((500,1))
    #Time = np.reshape(np.linspace(3.154e+7/4, 3.154e+7 * 150, 500),(500,1))
    #aFluence = (Flux * Time)
    #aEffectiveFluence = aFluence
    #combinedfluence = data.get_data(fluence_str) + lwr_data.get_data(fluence_str)
    #combinedflux = data.get_data(flux_str) + lwr_data.get_data(flux_str)
    #combinedeffectivefluence = data.get_data(eff_str) + lwr_data.get_data(eff_str)
    #normalize
    #TTM thought normalization was in the columns already? Renormalize for combined?
    #Flux = (np.log10(Flux) - np.min(combinedflux))/(np.max(combinedflux) - np.min(combinedflux))
    #Fluence  = (np.log10(aFluence) - np.min(combinedfluence))/(np.max(np.asarray(combinedfluence)) - np.min(combinedfluence))
    #EffectiveFluence = (np.log10(aEffectiveFluence) - np.min(combinedeffectivefluence))/(np.max(np.asarray(combinedeffectivefluence)) - np.min(combinedeffectivefluence))

    #for alloy in range(1, max(data.get_data("alloy_number"))[0] + 1):
    for alloy in range(1,5):
        print(alloy)
        data.remove_all_filters()
        data.add_inclusive_filter("alloy_number", '=', alloy)
        if len(data.get_x_data()) == 0: continue  # if alloy doesn't exist(x data is empty), then continue
        AlloyName = data.get_data("Alloy")[0][0]



        lwr_data.remove_all_filters()
        lwr_data.add_inclusive_filter("alloy_number", '=', alloy)
        #lwr_data.add_exclusive_filter(temp_str, '<>', 290)

        fluence_data = np.asarray(data.get_data(fluence_str)).ravel()
        predict_data = model.predict(data.get_x_data())
        points_data = np.asarray(data.get_y_data()).ravel()

        fluence_lwr = np.asarray(lwr_data.get_data(fluence_str)).ravel()
        predict_lwr = model.predict(lwr_data.get_x_data())
        points_lwr = np.asarray(lwr_data.get_y_data()).ravel()
        
        plt.figure()
        plt.hold(True)
        fig, ax = plt.subplots()
        matplotlib.rcParams.update({'font.size':18})
        ax.plot(fluence_lwr, predict_lwr,
                lw=3, color='#ffc04d', label="LWR prediction")
        ax.scatter(fluence_lwr, points_lwr,
                   lw=0, label="CD LWR data", color = '#7ec0ee')
        ax.scatter(fluence_data, points_data, lw=0, label='IVAR data',
                   color='black')

        #TTM remove else block; expt will have its own separate set of plots
        #and operate under the same block as above
        #else: 
        #    ax.scatter(np.log10(data.get_data("fluence n/cm2")), data.get_y_data(), lw=0, label='Expt IVAR data',
        #               color='green')
        #    if len(lwr_data.get_x_data()) == 0:
        #        Alloy = np.reshape(np.asarray(data.get_x_data())[0, 0:6], (1, 6)) * np.ones((500, 6))
        #        Xdata = np.concatenate([Alloy, EffectiveFluence], 1)
        #        ypredict = model.predict(Xdata)
        #        ax.plot(np.log10(aEffectiveFluence), ypredict, lw = 3, label='Model Prediction', color = 'orange')
        #    else:
        #        ax.plot(np.log10(lwr_data.get_data("fluence n/cm2")), model.predict(lwr_data.get_x_data()), lw=3,
        #                color='orange', label="Expt LWR prediction")



        plt.legend(loc = "upper left", fontsize=matplotlib.rcParams['font.size']) #data is sigmoid; 'best' can block data
        plt.title("{}({})".format(alloy,AlloyName))
        plt.xlabel("log(Fluence(n/cm$^{2}$))")
        plt.ylabel("$\Delta\sigma_{y}$ (MPa)")
        plt.savefig(savepath.format("%s_LWR" % ax.get_title()), dpi=200, bbox_inches='tight')
        plt.close()
        
        #print IVAR prediction plot
        plt.figure()
        matplotlib.rcParams.update({'font.size':18})
        #fluence is often not sorted; sort with predictions
        ivararr = np.array([fluence_data, predict_data]).transpose()
        ivar_tuples = tuple(map(tuple, ivararr))
        ivar_list = list(ivar_tuples).sort()
        ivararr_sorted = np.asarray(ivar_list)
        plt.plot(ivararr_sorted[:,0],ivararr_sorted[:,1],linestyle="None", linewidth=3,
                marker="o",
                color='#ffc04d', label="IVAR prediction")
        plt.plot(fluence_data,predict_data,linestyle="None", linewidth=3,
                marker="x",
                color='green', label="IVAR prediction unsorted test")
        plt.scatter(fluence_data, points_data, lw=0, label='IVAR data',
                   color='black')
        plt.legend(loc = "upper left", fontsize=matplotlib.rcParams['font.size']) #data is sigmoid; 'best' can block data
        plt.title("{}({})".format(alloy,AlloyName))
        plt.xlabel("log(Fluence(n/cm$^{2}$))")
        plt.ylabel("$\Delta\sigma_{y}$ (MPa)")
        plt.savefig(savepath.format("%s_IVAR" % ax.get_title()), dpi=200, bbox_inches='tight')
        plt.close()

        headerline = "logFluence LWR, Points LWR, Predicted LWR"
        myarray = np.array([fluence_lwr, points_lwr, predict_lwr]).transpose()
        ptools.array_to_csv("%s_LWR.csv" % AlloyName, headerline, myarray)
        headerline = "logFluence IVAR, Points IVAR, Predicted IVAR"
        myarray =np.array([fluence_data, points_data, predict_data]).transpose()
        ptools.array_to_csv("%s_IVAR.csv" % AlloyName, headerline, myarray)
