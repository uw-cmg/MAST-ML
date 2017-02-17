import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
import data_analysis.printout_tools as ptools
import matplotlib
from mean_error import mean_error

def execute(model, data, savepath, *args, **kwargs):
    atr2data=2 #old
    alloystr = "Alloy"
    codestr = "Data Set code"
    atr2data="ATR2"
    codestr = "dataset"
    alloystr= "alloy_number"
    data.remove_all_filters()
    data.add_exclusive_filter(codestr, '=', atr2data)
    model.fit(data.get_x_data(), data.get_y_data())

    atr1_alloys = [6,34,35,36,37,38]
    data.remove_all_filters()
    data.add_inclusive_filter(codestr, '=', atr2data)

    Ypredict = model.predict(data.get_x_data())
    Ypredict_array = np.asarray(Ypredict).ravel()
    overall_rms = np.sqrt(mean_squared_error(data.get_y_data(), Ypredict))
    overall_me = mean_error(Ypredict, data.get_y_data())
    
    alloys = np.asarray(data.get_data("alloy_number")).ravel()
    Ymeasured = np.asarray(data.get_y_data()).ravel()
    ptools.array_to_csv("ATRExtrapolation.csv","Alloy number,Measured Y,Predicted Y",np.array([alloys,Ymeasured,Ypredict_array]).transpose())
    Xerr = np.asarray(data.get_data("delta_sigma_y_MPa_uncertainty")).ravel()

    matplotlib.rcParams.update({'font.size':22})
    smallfont = 0.90 * matplotlib.rcParams['font.size']
    plt.figure()
    plt.hold(True)
    plt.errorbar(Ymeasured, Ypredict_array, xerr=Xerr, linewidth=1,
                linestyle = "None", color="red",
                markerfacecolor='red' , marker='o',
                markersize=10, markeredgecolor="None")
    plt.plot(plt.gca().get_ylim(), plt.gca().get_ylim(), ls="--", c=".3")
    plt.annotate("RMSE: %3.2f MPa" % overall_rms,xy=(0.05, 0.90), xycoords="axes fraction", fontsize=smallfont)
    plt.annotate("Mean error: %3.2f MPa" % overall_me ,xy=(0.05, 0.83), xycoords="axes fraction", fontsize=smallfont)
    plt.xlabel("Measured $\Delta\sigma_{y}$ (MPa)")
    plt.ylabel("Predicted $\Delta\sigma_{y}$ (MPa)")
    plt.axis('equal')
    plt.axis('square')
    plt.tight_layout()
    plt.savefig("ATR2simpleextrapolation.png")
    plt.hold(False)
    plt.close()
    #plt.scatter(data.get_y_data(), Ypredict, lw=0, color='#009AFF')
    #print(len(data.get_y_data()))

    data.remove_all_filters()
    for x in atr1_alloys:
        data.add_inclusive_filter(alloystr, '=', x)
    print(np.asarray(data.get_y_data()).shape)
    data.add_exclusive_filter(codestr, '<>', atr2data)
    print(np.asarray(data.get_y_data()).shape)
    Ypredict = model.predict(data.get_x_data())
    atr1_rms = np.sqrt(mean_squared_error(data.get_y_data(), Ypredict))
    plt.scatter(data.get_y_data(), Ypredict, lw = 0, color = '#03FF00', label = 'IVAR, ATR1, ATR2')
    #print(len(data.get_y_data()))

    data.remove_all_filters()
    data.add_inclusive_filter(alloystr,'<',60)
    for x in atr1_alloys:
        data.add_exclusive_filter(alloystr, '=', x)
    data.add_exclusive_filter(codestr, '<>', atr2data)
    Ypredict = model.predict(data.get_x_data())
    ivar_rms = np.sqrt(mean_squared_error(data.get_y_data(), Ypredict))
    plt.scatter(data.get_y_data(), Ypredict, lw=0, color='#FF0034', label = 'IVAR, ATR2')
    #print(len(data.get_y_data()))

    data.remove_all_filters()
    data.add_inclusive_filter(alloystr, '>', 59)
    data.add_exclusive_filter(codestr, '<>', atr2data)
    Ypredict = model.predict(data.get_x_data())
    atr2_rms = np.sqrt(mean_squared_error(data.get_y_data(), Ypredict))
    plt.scatter(data.get_y_data(), Ypredict, lw=0, color='#49CCFF', label = 'ATR2 only')
    #print(len(data.get_y_data()))

    plt.title('Predicting ATR-2, Training IVAR, ATR1')
    plt.figtext(.15, .84, 'Overall RMSE: {:.3f}'.format(overall_rms))
    plt.figtext(.15, .80, 'IVAR, ATR1, ATR2 alloys: {:.3f}'.format(atr1_rms))
    plt.figtext(.15, .76, 'IVAR, ATR2 alloys: {:.3f}'.format(ivar_rms))
    plt.figtext(.15, .72, 'ATR2 only alloys: {:.3f}'.format(atr2_rms))
    plt.legend(loc= 'lower right')
    plt.plot(plt.gca().get_ylim(), plt.gca().get_ylim(), ls="--", c=".3")
    plt.xlabel('Measured (MPa)')
    plt.ylabel('Predicted (MPa)')
    plt.savefig(savepath.format(plt.gca().get_title()), dpi=300, bbox_inches='tight')
    plt.close()
