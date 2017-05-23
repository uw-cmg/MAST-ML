import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import data_parser
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error

#todo in need of commenting and removal of descriptor hardcoding
def execute(model, data, savepath, *args, **kwargs):

    descriptors = ['Cu (At%)', 'Ni (At%)', 'Mn (At%)', 'P (At%)', 'Si (At%)', 'C (At%)', 'Temp (C)', 'log(fluence)',
                   'log(flux)']
    xlist = np.asarray(data.get_data(descriptors))

    model = model
    model.fit(data.get_x_data(), np.array(data.get_y_data()).ravel())
    error = model.predict(data.get_x_data()) - np.array(data.get_y_data()).ravel()

    for x in range(len(descriptors)):
        plt.scatter(xlist[:, x], error, color='black', s=10)
        xlim = plt.gca().get_xlim()
        plt.plot(xlim, (20, 20), ls="--", c=".3")
        plt.plot(xlim, (0, 0), ls="--", c=".3")
        plt.plot(xlim, (-20, -20), ls="--", c=".3")
        m, b = np.polyfit(np.reshape(xlist[:, x], len(xlist[:, x])), np.reshape(error, len(error)),1)  # line of best fit

        matplotlib.rcParams.update({'font.size': 15})
        plt.plot(xlist[:, x], m * xlist[:, x] + b, color='red')
        plt.figtext(.15, .83, 'y = ' + "{0:.6f}".format(m) + 'x + ' + "{0:.5f}".format(b), fontsize=14)
        plt.title('Error vs. {}'.format(descriptors[x]))
        plt.xlabel(descriptors[x])
        plt.ylabel('Predicted - Actual (Mpa)')
        plt.savefig(savepath.format(plt.gca().get_title()), dpi=200, bbox_inches='tight')
        plt.close()
