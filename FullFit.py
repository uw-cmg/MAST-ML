import matplotlib.pyplot as plt
import data_parser
import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error

def FullFit(model = KernelRidge(alpha= .00139, coef0=1, degree=3, gamma=.518, kernel='rbf', kernel_params=None), datapath = "../../DBTT_Data.csv", savepath = '../../{}.png'):
	data = data_parser.parse(datapath)
	data.set_x_features(["N(Cu)", "N(Ni)", "N(Mn)", "N(P)", "N(Si)","N( C )", "N(log(fluence)", "N(log(flux)", "N(Temp)"])
	data.set_y_feature("delta sigma")

	Ydata = np.asarray(data.get_y_data())
	Ydata_norm = (Ydata - np.mean(Ydata))/np.std(Ydata)

	IVARindices = np.linspace(0,1463,1464).astype(int)
	IVARplusindices = np.linspace(1464,1506,43).astype(int)

	model = model

	# Train the model using the training sets
	model.fit(data.get_x_data(), Ydata_norm)

	Ypredict_norm = model.predict(data.get_x_data())
	Ypredict = Ypredict_norm*np.std(Ydata)+np.mean(Ydata)

	# calculate rms
	rms = np.sqrt(mean_squared_error(Ypredict,Ydata))
	IVAR_rms = np.sqrt(mean_squared_error(Ypredict[IVARindices],Ydata[IVARindices]))
	IVARplus_rms = np.sqrt(mean_squared_error(Ypredict[IVARplusindices],Ydata[IVARplusindices]))
	print('RMS: %.5f, IVAR RMS: %.5f, IVAR+ RMS: %.5f' %(rms, IVAR_rms, IVARplus_rms))

	# graph outputs
	plt.scatter(Ydata[IVARindices], Ypredict[IVARindices], s = 10, color='black', label = 'IVAR')
	plt.legend(loc = 4)
	plt.scatter(Ydata[IVARplusindices], Ypredict[IVARplusindices], s = 10, color='red', label = 'IVAR+')
	plt.legend(loc = 4)
	plt.plot(plt.gca().get_ylim(), plt.gca().get_ylim(), ls="--", c=".3")
	plt.xlabel('Measured (MPa)')
	plt.ylabel('Predicted (MPa)')
	plt.title('Full Fit')
	plt.figtext(.15,.83,'Overall RMS: %.4f' %(rms), fontsize = 14)
	plt.figtext(.15,.77,'IVAR RMS: %.4f' %(IVAR_rms), fontsize = 14)
	plt.figtext(.15,.71,'IVAR+ RMS: %.4f' %(IVARplus_rms), fontsize = 14)
	plt.savefig(savepath.format(plt.gca().get_title()), dpi = 200, bbox_inches='tight')
	plt.show()
	plt.close()


