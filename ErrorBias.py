import matplotlib.pyplot as plt
import numpy as np
import data_parser
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error

def getModel():
    return KernelRidge(alpha= .00139, coef0=1, degree=3, gamma=.518, kernel='rbf', kernel_params=None)

data = data_parser.parse("../../DBTT_Data.csv")
data.set_x_features(["N(Cu)", "N(Ni)", "N(Mn)", "N(P)", "N(Si)","N( C )", "N(log(fluence)", "N(log(flux)", "N(Temp)"])
data.set_y_feature("delta sigma")

descriptors = ['Cu (At%)','Ni (At%)','Mn (At%)','P (At%)', 'Si (At%)', 'C (At%)', 'log(fluence)', 'log(flux)', 'Temp (C)']
Xlist = np.asarray(data.get_data(descriptors))

model = getModel()
model.fit(data.get_x_data(), data.get_y_data())
Error = model.predict(data.get_x_data()) - data.get_y_data()

for x in range(len(descriptors)):
    plt.scatter(Xlist[:,x], Error, color='black')
    xlim = plt.gca().get_xlim()
    plt.plot(xlim, (20, 20), ls="--", c=".3")
    plt.plot(xlim, (0,0), ls="--", c=".3")
    plt.plot(xlim, (-20, -20), ls="--", c=".3")
    m, b = np.polyfit(np.reshape(Xlist[:,x], len(Xlist[:,x])), np.reshape(Error, len(Error)), 1) #line of best fit
    plt.plot(Xlist[:,x], m*Xlist[:,x]+b, color = 'red')
    plt.figtext(.15,.83,'y = ' + "{0:.6f}".format(m) + 'x + ' + "{0:.5f}".format(b), fontsize = 14)
    plt.title('Error vs. {}'.format(descriptors[x]))
    plt.xlabel(descriptors[x])
    plt.ylabel('Predicted - Actual (Mpa)')
    plt.savefig('../../{}.png'.format(plt.gca().get_title()), dpi = 200, bbox_inches='tight')               
    plt.show()


