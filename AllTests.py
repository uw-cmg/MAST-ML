import KFold_CV
import LeaveOutAlloyCV
import FullFit
import FluenceFluxExtrapolation
import DescriptorImportance
import ErrorBias
import timeit

start = timeit.default_timer()
# things to change before running the codes

# model to use
from sklearn.kernel_ridge import KernelRidge
model = KernelRidge(alpha=.00139, coef0=1, degree=3, gamma=.518, kernel='rbf', kernel_params=None)

# file paths
datapath = "../../DBTT_Data.csv"  # path to your data
savepath = "../../graphs/{}.png"  # where you want output graphs to be saved

print("K-Fold CV:")
KFold_CV.cv(model, datapath, savepath)  # also has parameters num_folds (default is 5) and num_runs (default is 200)
stop1 = timeit.default_timer()

print("\nLeave out alloy CV:")
LeaveOutAlloyCV.LOACV(model, datapath, savepath)
stop2 = timeit.default_timer()
print("\nFull Fit:")
FullFit.FullFit(model, datapath, savepath)
stop3 = timeit.default_timer()
print("\nFluence and Flux Extrapolation:")
FluenceFluxExtrapolation.FlFxExt(model, datapath, savepath)
stop4 = timeit.default_timer()
print("\nError Bias:")
ErrorBias.ErrBias(model, datapath, savepath)
stop5 = timeit.default_timer()
print("\nDescriptor Importance:")
DescriptorImportance.DesImp(model, datapath, savepath)
stop6 = timeit.default_timer()

print (stop1 - start)
print (stop2 - stop1)
print (stop3 - stop2)
print (stop4 - stop3)
print (stop5 - stop4)
print (stop6 - stop5)
print (stop6 - start)