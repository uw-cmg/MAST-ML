import KFold_CV
import LeaveOutAlloyCV
import FullFit
import FluenceFluxExtrapolation
import DescriptorImportance
import ErrorBias
import configuration_parser
import importlib
# things to change before running the codes

# model to use
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

config = configuration_parser.parse()

model = importlib.import_module(config.get('all_test', 'model')).get()
datapath = config.get('all_test', 'data_path')
savepath = config.get('all_test', 'save_path')

Ydata = config.get('all_test', 'Y')

print("K-Fold CV:")
KFold_CV.cv(model, datapath, savepath, Y = Ydata)  # also has parameters num_folds (default is 5) and num_runs (default is 200)

print("\nLeave out alloy CV:")
LeaveOutAlloyCV.loacv(model, datapath, savepath, Y = Ydata)

print("\nFull Fit:")
FullFit.fullfit(model, datapath, savepath, Y = Ydata)

print("\nFluence and Flux Extrapolation:")
FluenceFluxExtrapolation.flfxex(model, datapath, savepath, Y = Ydata)

print("\nError Bias:")
ErrorBias.errbias(model, datapath, savepath, Y = Ydata)


print("\nDescriptor Importance:")
DescriptorImportance.desimp(model, datapath, savepath, Y = Ydata)


