import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import data_parser
from sklearn.metrics import mean_squared_error
from sklearn.kernel_ridge import KernelRidge


def flfxex(model=KernelRidge(alpha=.00139, coef0=1, degree=3, gamma=.518, kernel='rbf', kernel_params=None),
            datapath="../../DBTT_Data.csv", savepath='../../{}.png',
            X=["N(Cu)", "N(Ni)", "N(Mn)", "N(P)", "N(Si)", "N( C )", "N(log(fluence)", "N(log(flux)", "N(Temp)"],
            Y="delta sigma"):

    data = data_parser.parse(datapath)
    data.set_x_features(X)
    data.set_y_feature(Y)

    fluence_divisions = [3.3E18, 3.3E19, 3.3E20]
    flux_divisions = [5e11,2e11,1e11]

    fig, ax = plt.subplots(1,3, figsize = (30,10))
    for x in range(len(fluence_divisions)):
        model = model
        data.remove_all_filters()
        data.add_inclusive_filter("fluence n/cm2", '<', fluence_divisions[x])
        l_train = len(data.get_y_data())
        model.fit(data.get_x_data(), data.get_y_data().ravel())

        data.remove_all_filters()
        data.add_inclusive_filter("fluence n/cm2", '>=', fluence_divisions[x])
        l_test = len(data.get_y_data())
        Ypredict = model.predict(data.get_x_data())
        RMSE = np.sqrt(mean_squared_error(Ypredict, data.get_y_data().ravel()))

        matplotlib.rcParams.update({'font.size': 26})
        ax[x].scatter(data.get_y_data(), Ypredict, color='black', s=10)
        ax[x].plot(ax[x].get_ylim(), ax[x].get_ylim(), ls="--", c=".3")
        ax[x].set_xlabel('Measured ∆sigma (Mpa)')
        ax[x].set_ylabel('Predicted ∆sigma (Mpa)')
        ax[x].set_title('Testing Fluence > {}'.format(fluence_divisions[x]))
        ax[x].text(.1, .88, 'RMSE: {:.3f}'.format(RMSE),fontsize = 30, transform=ax[x].transAxes)
        ax[x].text(.1, .83, 'Train: {}, Test: {}'.format(l_train, l_test), transform=ax[x].transAxes)

    fig.tight_layout()
    plt.subplots_adjust(bottom = .2)
    fig.savefig(savepath.format("fluence_extrapolation"), dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()

    fig, ax = plt.subplots(1, 3, figsize=(30, 10))
    for x in range(len(flux_divisions)):
        model = model
        data.remove_all_filters()
        data.add_inclusive_filter("flux n/cm2/s", '>', flux_divisions[x])
        l_train = len(data.get_y_data())
        model.fit(data.get_x_data(), data.get_y_data().ravel())

        data.remove_all_filters()
        data.add_inclusive_filter("flux n/cm2/s", '<=', flux_divisions[x])
        l_test = len(data.get_y_data())
        Ypredict = model.predict(data.get_x_data())
        RMSE = np.sqrt(mean_squared_error(Ypredict, data.get_y_data().ravel()))

        matplotlib.rcParams.update({'font.size': 26})
        ax[x].scatter(data.get_y_data(), Ypredict, color='black', s=10)
        ax[x].plot(ax[x].get_ylim(), ax[x].get_ylim(), ls="--", c=".3")
        ax[x].set_xlabel('Measured ∆sigma (Mpa)')
        ax[x].set_ylabel('Predicted ∆sigma (Mpa)')
        ax[x].set_title('Testing Flux < {:.0e}'.format(flux_divisions[x]))
        ax[x].text(.1, .88, 'RMSE: {:.3f}'.format(RMSE), fontsize=30, transform=ax[x].transAxes)
        ax[x].text(.1, .83, 'Train: {}, Test: {}'.format(l_train, l_test), transform=ax[x].transAxes)

    fig.tight_layout()
    plt.subplots_adjust(bottom=.2)
    fig.savefig(savepath.format("flux_extrapolation"), dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()