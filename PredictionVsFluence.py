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
    if not "temp_filter" in kwargs.keys():
        temp_filter = None
    else:
        temp_filter = int(kwargs["temp_filter"]) #int matches data
    Xdata = np.asarray(data.get_x_data())
    Ydata = np.asarray(data.get_y_data()).ravel()

    model.fit(Xdata, Ydata)
    #note that temp_filter only affects the plotting, not the model fitting

    fluence_str = "log(fluence_n_cm2)"
    flux_str = "log(flux_n_cm2_sec)"
    eff_str = "log(eff fl 100p=26)"
    temp_str = "temperature_C"

    #Creating a renormalized effective fluence column, renormalized
    #based on the min/max from both the IVAR and the LWR data
    #This column will only be used for the LWR predictions if there
    #is no LWR data.
    Flux = 3e10 * np.ones((500,1))
    Time = np.reshape(np.linspace(3.154e+7/4, 3.154e+7 * 150, 500),(500,1))
    aFluence = (Flux * Time)
    aEffectiveFluence = aFluence
    combinedfluence = data.get_data(fluence_str) + lwr_data.get_data(fluence_str)
    combinedflux = data.get_data(flux_str) + lwr_data.get_data(flux_str)
    combinedeffectivefluence = data.get_data(eff_str) + lwr_data.get_data(eff_str)
    #normalize
    Flux = (np.log10(Flux) - np.min(combinedflux))/(np.max(combinedflux) - np.min(combinedflux))
    Fluence  = (np.log10(aFluence) - np.min(combinedfluence))/(np.max(np.asarray(combinedfluence)) - np.min(combinedfluence))
    EffectiveFluence = (np.log10(aEffectiveFluence) - np.min(combinedeffectivefluence))/(np.max(np.asarray(combinedeffectivefluence)) - np.min(combinedeffectivefluence))
    mock_xdata = np.reshape(np.asarray(data.get_x_data())[0, 0:6], (1, 6)) * np.ones((500, 6))
    mock_condition_set = np.concatenate([mock_xdata, EffectiveFluence], 1)

    #for alloy in range(1,5): #restrict range for testing
    for alloy in range(1, max(data.get_data("alloy_number"))[0] + 1):
        print(alloy)
        data.remove_all_filters()
        data.add_inclusive_filter("alloy_number", '=', alloy)
        print(np.asarray(data.get_x_data()).shape)
        if temp_filter == None:
            pass
        else:
            data.add_exclusive_filter(temp_str,'<>',temp_filter)    
        print(np.asarray(data.get_x_data()).shape)
        if len(data.get_x_data()) == 0: continue  # if alloy doesn't exist(x data is empty), then continue
        AlloyName = data.get_data("Alloy")[0][0]

        fluence_data = np.asarray(data.get_data(fluence_str)).ravel()
        predict_data = model.predict(data.get_x_data())
        points_data = np.asarray(data.get_y_data()).ravel()
        eff_fluence_data = np.asarray(data.get_data(eff_str)).ravel()
        
        #print IVAR prediction plot
        plt.figure()
        matplotlib.rcParams.update({'font.size':18})
        #fluence is often not sorted; sort with predictions
        ivararr = np.array([eff_fluence_data, predict_data]).transpose()
        ivar_tuples = tuple(map(tuple, ivararr))
        ivar_list = list(ivar_tuples)
        ivar_list.sort()
        ivararr_sorted = np.asarray(ivar_list)
        plt.plot(ivararr_sorted[:,0],ivararr_sorted[:,1],linestyle="-", 
                linewidth=3,
                marker=None,
                color='#ffc04d', label="IVAR prediction")
        #plt.plot(fluence_data,predict_data,linestyle="None", linewidth=3,
        #        marker="x", markersize=10,
        #        color='green', label="IVAR prediction unsorted test")
        plt.scatter(eff_fluence_data, points_data, lw=0, label='IVAR data',
                   color='black')
        plt.legend(loc = "upper left", fontsize=matplotlib.rcParams['font.size']) #data is sigmoid; 'best' can block data
        alloystr = "{}({})".format(alloy,AlloyName)
        plt.title(alloystr)
        plt.xlabel("log(Eff Fluence(n/cm$^{2}$))")
        plt.ylabel("$\Delta\sigma_{y}$ (MPa)")
        plt.savefig(savepath.format("%s_IVAR" % alloystr), dpi=200, bbox_inches='tight')
        plt.close()
        
        headerline = "logEffFluence IVAR, Points IVAR, Predicted IVAR"
        myarray =np.array([eff_fluence_data, points_data, predict_data]).transpose()
        ptools.array_to_csv("%s_IVAR.csv" % AlloyName, headerline, myarray)

        #LWR set
        lwr_data.remove_all_filters()
        lwr_data.add_inclusive_filter("alloy_number", '=', alloy)
        if temp_filter == None:
            pass
        else:
            lwr_data.add_exclusive_filter(temp_str, '<>', temp_filter)

        if len(lwr_data.get_x_data()) == 0:
            plot_lwr_points = False
            predict_lwr = model.predict(mock_condition_set)
            eff_fluence_lwr = EffectiveFluence
            points_lwr = np.zeros(len(eff_fluence_lwr)) #empty for printout
        
        else:
            plot_lwr_points = True
            fluence_lwr = np.asarray(lwr_data.get_data(fluence_str)).ravel()
            predict_lwr = model.predict(lwr_data.get_x_data())
            points_lwr = np.asarray(lwr_data.get_y_data()).ravel()
            eff_fluence_lwr = np.asarray(lwr_data.get_data(eff_str)).ravel()
        
        plt.figure()
        plt.hold(True)
        fig, ax = plt.subplots()
        matplotlib.rcParams.update({'font.size':18})
        ax.plot(eff_fluence_lwr, predict_lwr,
                lw=3, color='#ffc04d', label="LWR prediction")
        ax.scatter(eff_fluence_data, points_data, lw=0, label='IVAR data',
                   color='black')   
        if plot_lwr_points:
            ax.scatter(eff_fluence_lwr, points_lwr,
                   lw=0, label="CD LWR data", color = '#7ec0ee')

        plt.legend(loc = "upper left", fontsize=matplotlib.rcParams['font.size']) #data is sigmoid; 'best' can block data
        plt.title("{}({})".format(alloy,AlloyName))
        plt.xlabel("log(Eff Fluence(n/cm$^{2}$))")
        plt.ylabel("$\Delta\sigma_{y}$ (MPa)")
        plt.savefig(savepath.format("%s_LWR" % ax.get_title()), dpi=200, bbox_inches='tight')
        plt.close()
        

        headerline = "logEffFluence LWR, Points LWR, Predicted LWR"
        myarray = np.array([eff_fluence_lwr, points_lwr, predict_lwr]).transpose()
        ptools.array_to_csv("%s_LWR.csv" % AlloyName, headerline, myarray)
    return
