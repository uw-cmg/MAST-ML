import matplotlib.pyplot as plt
import data_parser
import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error


def execute(model, data, savepath, lwr_datapath = "CD_LWR_clean"):

    lwr_data = data_parser.parse(lwr_datapath)
    lwr_data.set_x_features(["N(Cu)", "N(Ni)", "N(Mn)", "N(P)","N(Si)", "N( C )", "N(log(fluence)", "N(log(flux)", "N(Temp)"])
    lwr_data.set_y_feature("CD delta sigma")
    Ydata = np.asarray(lwr_data.get_y_data()).ravel()

    # Train the model using the training sets
    model.fit(data.get_x_data(), np.asarray(data.get_y_data()).ravel())

    Ypredict = model.predict(lwr_data.get_x_data())

    # calculate rms
    rms = np.sqrt(mean_squared_error(Ypredict, Ydata))
    IVAR_rms = np.sqrt(mean_squared_error(Ypredict[IVARindices], Ydata[IVARindices]))
    IVARplus_rms = np.sqrt(mean_squared_error(Ypredict[IVARplusindices], Ydata[IVARplusindices]))
    print('RMS: %.5f, IVAR RMS: %.5f, IVAR+ RMS: %.5f' % (rms, IVAR_rms, IVARplus_rms))

    # graph outputs
    plt.figure(1)
    plt.scatter(Ydata, Ypredict, s=10, color='black')
    plt.plot(plt.gca().get_ylim(), plt.gca().get_ylim(), ls="--", c=".3")
    plt.xlabel('Measured (MPa)')
    plt.ylabel('Predicted (MPa)')
    plt.title('Extrapolation to LWR')
    plt.figtext(.15, .83, 'Overall RMS: %.4f' % (rms), fontsize=14)
    plt.figtext(.15, .77, 'IVAR RMS: %.4f' % (IVAR_rms), fontsize=14)
    plt.figtext(.15, .71, 'IVAR+ RMS: %.4f' % (IVARplus_rms), fontsize=14)
    plt.savefig(savepath.format(plt.gca().get_title()), dpi=200, bbox_inches='tight')

    plt.clf()
    plt.close()
