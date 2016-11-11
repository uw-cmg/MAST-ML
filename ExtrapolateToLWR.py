import matplotlib.pyplot as plt
import data_parser
import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error
from mean_error import mean_error
import matplotlib.cm as cmx
import matplotlib.colors as colors
import csv

def get_cmap(N):
    '''Returns a function that maps each index in 0, 1, ... N-1 to a distinct
    RGB color.'''
    color_norm  = colors.Normalize(vmin=0, vmax=N-1)
    scalar_map = cmx.ScalarMappable(norm=color_norm, cmap='hsv')
    def map_index_to_rgb_color(index):
        return colors.rgb2hex(scalar_map.to_rgba(index))
    return map_index_to_rgb_color

'''
Compare CD predictions of ∆sigma at LWR conditions  w/ model's prediction when fit to available experimental data.
Special interest is paid to large X points (often high fluence) and 60-100 year points (extension goal)

Plots model's predictions(CD predictions(lwr_data)).
model is trained to data, which is in the domain of IVAR, IVAR+/ATR1 (assorted fluxes and fluences)
lwr_data is data in the domain of LWR conditions (low flux, high fluence)
'''

def execute(model, data, savepath, lwr_data, *args, **kwargs):

    if(data.y_feature == "delta sigma"):
        print("Must set Y to be CD Delta Sigma or EONY Delta Sigma for Extrapolating to LWR")
        return

    trainX = np.asarray(data.get_x_data())
    trainY = np.asarray(data.get_y_data()).ravel()

    # Train the model using the training sets
    model.fit(trainX, trainY)

    alloy_list = []
    rms_list = []
    fig, ax = plt.subplots(1, 2, figsize = (11,5))
    cmap = get_cmap(60)
    for alloy in range(1,60):

        lwr_data.remove_all_filters()
        lwr_data.add_inclusive_filter("Alloy", '=', alloy)

        if len(lwr_data.get_x_data()) == 0: continue

        alloy_list.append(alloy)
        testX = np.asarray(lwr_data.get_x_data())
        Ypredict = model.predict(testX)
        Yactual = np.asarray(lwr_data.get_y_data()).ravel()
        rms = np.sqrt(mean_squared_error(Ypredict, Yactual))
        rms_list.append(rms)
        if rms > 200:
            ax[0].scatter(Yactual, Ypredict, s= 5, c = cmap(alloy), label= alloy, lw = 0)
        else: ax[0].scatter(Yactual, Ypredict, s = 5, c = 'black', lw = 0)

        lwr_data.add_exclusive_filter("Time(Years)", '<', 60)
        lwr_data.add_exclusive_filter("Time(Years)", '>', 100)

        testX = np.asarray(lwr_data.get_x_data())
        Ypredict = model.predict(testX)
        Yactual = np.asarray(lwr_data.get_y_data()).ravel()
        rms = np.sqrt(mean_squared_error(Ypredict, Yactual))
        ax[1].scatter(Yactual, Ypredict, s=5, c='black', lw=0)

    lwr_data.remove_all_filters()
    overall_y_data = np.asarray(lwr_data.get_y_data()).ravel()
    overall_y_prediction = model.predict(lwr_data.get_x_data())
    overall_rms = np.sqrt(mean_squared_error(overall_y_prediction, overall_y_data))
    overall_me = mean_error(overall_y_prediction, overall_y_data)
    lwr_data.add_exclusive_filter("CD delta sigma", '<', 200)
    over200_rms = np.sqrt(mean_squared_error(np.asarray(lwr_data.get_y_data()).ravel(), model.predict(lwr_data.get_x_data())))
    lwr_data.remove_all_filters()
    lwr_data.add_exclusive_filter("Time(Years)", '<', 60)
    lwr_data.add_exclusive_filter("Time(Years)", '>', 100)
    y_data_60_100 = np.asarray(lwr_data.get_y_data()).ravel()
    y_prediction_60_100 = model.predict(lwr_data.get_x_data())
    lwr_rms = np.sqrt(mean_squared_error(y_data_60_100, y_prediction_60_100))
    lwr_me = mean_error(y_prediction_60_100,y_data_60_100)
    lwr_data.add_exclusive_filter("CD delta sigma", '<', 200)
    lwr_over200_rms = np.sqrt(
        mean_squared_error(np.asarray(lwr_data.get_y_data()).ravel(), model.predict(lwr_data.get_x_data())))

    ax[0].legend()
    ax[0].plot(plt.gca().get_xlim(), plt.gca().get_xlim(), ls="--", c=".3")
    ax[0].set_xlabel('{} (MPa)'.format(data.y_feature))
    ax[0].set_ylabel('Model Predicted (MPa)')
    ax[0].set_title('Overall Extrapolation to LWR')
    ax[0].text(.05, .88, 'RMSE: %.2f MPa' % (overall_rms), fontsize=14, transform=ax[0].transAxes)
    ax[0].text(.05, .82, 'CD >200MPa RMSE: %.2f MPa' % (over200_rms), fontsize=14, transform=ax[0].transAxes)
    ax[0].text(.05, .76, 'Mean Error: %.2f MPa' % (overall_me), fontsize=14, transform=ax[0].transAxes)
    ax[1].plot(plt.gca().get_xlim(), plt.gca().get_xlim(), ls="--", c=".3")
    ax[1].set_xlabel('{} (MPa)'.format(data.y_feature))
    ax[1].set_ylabel('Model Predicted (MPa)')
    ax[1].set_title('60-100 year Extrapolation')
    ax[1].text(.05, .88, 'RMSE: %.2f MPa' % (lwr_rms), fontsize=14, transform=ax[1].transAxes)
    ax[1].text(.05, .82, 'CD >200MPa RMSE: %.2f MPa' % (lwr_over200_rms), fontsize=14, transform=ax[1].transAxes)
    ax[1].text(.05, .76, 'Mean Error: %.2f MPa' % (lwr_me), fontsize=14, transform=ax[1].transAxes)
    fig.tight_layout()
    fig.savefig(savepath.format(plt.gca().get_title()), dpi=300, bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 4))
    plt.xticks(np.arange(0, max(alloy_list) + 1, 5))
    ax.scatter(alloy_list, rms_list, s = 10, color = 'black')
    ax.plot((0, 59), (0, 0), ls="--", c=".3")
    ax.set_xlabel('Alloy')
    ax.set_ylabel('RMSE')
    ax.set_title('Extrapolate to LWR per Alloy')
    for x in np.argsort(rms_list)[-5:]:
        ax.annotate(s=alloy_list[x], xy=(alloy_list[x], rms_list[x]))
    plt.savefig(savepath.format("Extrapolate to LWR"), dpi = 300, bbox_inches='tight')
    plt.close()

    with open(savepath.replace(".png", "").format("Extrapolate to LWR.csv"), 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        x = ["Actual Data, Overall", "Prediction, Overall"]
        writer.writerow(x)
        for i in zip(overall_y_data,overall_y_prediction):
            writer.writerow(i)
        writer.writerow(["60-100 Year Data", "60-100 Year Prediction"])
        for j in zip(y_data_60_100,y_prediction_60_100):
            writer.writerow(j)

    with open(savepath.replace(".png", "").format("Extrapolate to LWR per Alloy RMSE.csv"), 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        x = ["Alloy", "RMSE"]
        writer.writerow(x)
        for i in zip(alloy_list, rms_list):
            writer.writerow(i)
