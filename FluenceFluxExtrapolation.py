import matplotlib.pyplot as plt
import numpy as np
import data_parser
from sklearn.metrics import mean_squared_error
from sklearn.kernel_ridge import KernelRidge

def getModel():
    return KernelRidge(alpha= .00139, coef0=1, degree=3, gamma=.518, kernel='rbf', kernel_params=None)

data = data_parser.parse("../../DBTT_data.csv")
data.set_x_features(["N(Cu)", "N(Ni)", "N(Mn)", "N(P)", "N(Si)","N( C )", "N(log(fluence)", "N(log(flux)", "N(Temp)"])
data.set_y_feature("delta sigma")

fluence_divisions = [3.3E18, 3.3E19, 3.3E20]
flux_divisions = [5.3e11, 4e12, 3e13]

for x in fluence_divisions:
    model = getModel()
    data.remove_filter()
    data.add_filter("fluence n/cm2", '<', x)
    l_train = len(data.get_y_data())
    model.fit(data.get_x_data(), data.get_y_data())

    data.remove_filter()
    data.add_filter("fluence n/cm2", '>=', x)
    l_test = len(data.get_y_data())
    Ypredict = model.predict(data.get_x_data())
    RMSE = np.sqrt(mean_squared_error(Ypredict, data.get_y_data()))

    plt.scatter(data.get_y_data(), Ypredict, color = 'black')
    plt.plot(plt.gca().get_ylim(), plt.gca().get_ylim(), ls="--", c=".3")
    plt.xlabel('Measured ∆sigma (Mpa)')
    plt.ylabel('Predicted ∆sigma (Mpa)')
    plt.title('Testing Fluence > {}'.format(x))
    plt.figtext(.15,.83,'RMSE: {:.3f}'.format(RMSE), fontsize = 14)
    plt.figtext(.15,.78,'Train: {}, Test: {}'.format(l_train, l_test), fontsize = 13)

    plt.savefig('../../{}.png'.format(plt.gca().get_title()), dpi = 200, bbox_inches='tight')
    plt.show()

for x in flux_divisions:
    model = getModel()
    data.remove_filter()
    data.add_filter("flux n/cm2/s", '<', x)
    l_train = len(data.get_y_data())
    model.fit(data.get_x_data(), data.get_y_data())

    data.remove_filter()
    data.add_filter("flux n/cm2/s", '>=', x)
    l_test = len(data.get_y_data())
    Ypredict = model.predict(data.get_x_data())
    RMSE = np.sqrt(mean_squared_error(Ypredict, data.get_y_data()))

    plt.scatter(data.get_y_data(), Ypredict, color = 'black')
    plt.plot(plt.gca().get_ylim(), plt.gca().get_ylim(), ls="--", c=".3")
    plt.xlabel('Measured ∆sigma (Mpa)')
    plt.ylabel('Predicted ∆sigma (Mpa)')
    plt.title('Testing Flux > {:.1E}'.format(x))
    plt.figtext(.15,.83,'RMSE: {:.2f}'.format(RMSE), fontsize = 14)
    plt.figtext(.15,.78,'Train: {}, Test: {}'.format(l_train, l_test), fontsize = 13)

    plt.savefig('../../{}.png'.format(plt.gca().get_title()), dpi = 200, bbox_inches='tight')
    plt.show()
