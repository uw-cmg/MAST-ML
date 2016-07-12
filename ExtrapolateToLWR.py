import matplotlib.pyplot as plt
import data_parser
import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error
import matplotlib.cm as cmx
import matplotlib.colors as colors

def get_cmap(N):
    '''Returns a function that maps each index in 0, 1, ... N-1 to a distinct
    RGB color.'''
    color_norm  = colors.Normalize(vmin=0, vmax=N-1)
    scalar_map = cmx.ScalarMappable(norm=color_norm, cmap='hsv')
    def map_index_to_rgb_color(index):
        return scalar_map.to_rgba(index)
    return map_index_to_rgb_color
    
def execute(model, data, savepath, lwr_datapath = "CD_LWR_clean.csv"):

    X = ["N(Cu)", "N(Ni)", "N(Mn)", "N(P)","N(Si)", "N( C )", "N(log(fluence)", "N(log(flux)", "N(Temp)"]
    Y = "CD delta sigma"
    trainX = np.asarray(data.get_x_data())
    trainY = np.asarray(data.get_y_data()).ravel()

    lwr_data = data_parser.parse(lwr_datapath)
    lwr_data.set_x_features(X)
    lwr_data.set_y_feature(Y)
    

    # Train the model using the training sets
    model.fit(trainX, trainY)

    alloy_list = []
    rms_list = []
    fig, ax = plt.subplots()
    cmap = get_cmap(60)
    for alloy in range(1,60):

        lwr_data.remove_all_filters()
        lwr_data.add_inclusive_filter("Alloy", '=', alloy)
        if len(lwr_data.get_x_data()) == 0: continue

        alloy_list.append(alloy)
        testX = np.asarray(lwr_data.get_x_data())
        Ypredict = model.predict(testX)
        rms = np.sqrt(mean_squared_error(Ypredict, lwr_data.get_y_data()))
        rms_list.append(rms)
        if rms > 40:
            ax.scatter(lwr_data.get_y_data(), Ypredict, s=5, c = cmap(alloy), label= alloy, lw = 0)
        else: ax.scatter(lwr_data.get_y_data(), Ypredict, s = 5, c = 'black', lw = 0)


    lwr_data.remove_all_filters()
    rms = np.sqrt(mean_squared_error(model.predict(lwr_data.get_x_data()), lwr_data.get_y_data()))

    ax.legend()
    ax.plot(plt.gca().get_ylim(), plt.gca().get_ylim(), ls="--", c=".3")
    ax.set_xlabel(Y)
    ax.set_ylabel('Model Predicted (MPa)')
    ax.set_title('Extrapolate to LWR')
    plt.figtext(.15, .83, 'RMS: %.4f' % (rms), fontsize=14)
    plt.savefig(savepath.format(plt.gca().get_title()), dpi=300, bbox_inches='tight')
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
    plt.savefig(savepath.format(plt.gca().get_title()), dpi = 300, bbox_inches='tight')
    plt.close()