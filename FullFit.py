import matplotlib.pyplot as plt
import data_parser
import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error


def execute(model, data, savepath, *args):
    # Train the model using the training sets
    model.fit(data.get_x_data(), np.asarray(data.get_y_data()).ravel())
    overall_rms = np.sqrt(mean_squared_error(model.predict(data.get_x_data()), np.asarray(data.get_y_data()).ravel()))
    datasets = ['IVAR', 'ATR-1', 'ATR-2']
    colors = ['#BCBDBD', '#009AFF', '#FF0A09']
    fig, ax = plt.subplots()
    #calculate rms for each dataset
    for dataset in range(max(np.asarray(data.get_data("Data Set code")).ravel()) + 1):
        data.remove_all_filters()
        data.add_inclusive_filter("Data Set code", '=', dataset)
        Ypredict = model.predict(data.get_x_data())
        Ydata = np.asarray(data.get_y_data()).ravel()
        # calculate rms
        rms = np.sqrt(mean_squared_error(Ypredict, Ydata))
        # graph outputs
        ax.scatter(Ydata, Ypredict, s=7, color=colors[dataset], label= datasets[dataset], lw = 0)
        ax.text(.05, .83 - .05*dataset, '{} RMS: {:.3f}'.format(datasets[dataset],rms), fontsize=14, transform=ax.transAxes)

    ax.legend()
    ax.plot(ax.get_ylim(), ax.get_ylim(), ls="--", c=".3")
    ax.set_xlabel('Measured (MPa)')
    ax.set_ylabel('Predicted (MPa)')
    ax.set_title('Full Fit')
    ax.text(.05, .88, 'Overall RMS: %.4f' % (overall_rms), fontsize=14, transform=ax.transAxes)
    fig.savefig(savepath.format(ax.get_title()), dpi=300, bbox_inches='tight')
    plt.clf()
    plt.close()
