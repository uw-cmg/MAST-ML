import os
import matplotlib
import numpy as np
import data_parser
import matplotlib.pyplot as plt
from mean_error import mean_error
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error
import data_analysis.printout_tools as ptools
import plot_data.plot_predicted_vs_measured as plotpm

def execute(model, data, savepath, lwr_data="",
            xlabel="Measured",
            ylabel="Predicted",
            numericlabelfield=None,
            stepsize=1,
            *args, **kwargs):
    """Basic cross validation
        Args:
            model, data, savepath, lwr_data: see AllTests.py
            xlabel <str>: x-axis label for plot
            ylabel <str>: y-axis label for plot
            numericlabelfield <str>: Field for label data (numeric only)
    """
    stepsize = float(stepsize)

    # get data
    Ydata = np.array(data.get_y_data()).ravel()
    Xdata = np.array(data.get_x_data())
    dlen = len(Xdata)
    indices = np.arange(0, dlen)

    print("Using leave one out.")
    cvmodel = LeaveOneOut()

    Run_Y_Pred = np.empty(dlen)
    Run_Abs_Err = np.empty(dlen)
    Run_Y_Pred.fill(np.nan)
    Run_Abs_Err.fill(np.nan)
    # split into testing and training sets
    ntest=0
    ntests = len(indices)
    for train, test in cvmodel.split(indices):
        ntest = ntest + 1
        print("Test %i/%i" % (ntest,ntests))
        X_train, X_test = Xdata[train], Xdata[test]
        Y_train, Y_test = Ydata[train], Ydata[test]
        # train on training sets
        model.fit(X_train, Y_train)
        Y_test_Pred = model.predict(X_test)
        Run_Y_Pred[test] = Y_test_Pred
        Run_Abs_Err[test] = np.absolute(Y_test - Y_test_Pred)

    RMSE = np.sqrt(np.mean(np.power(Run_Abs_Err, 2.0)))
    maxErr = max(Run_Abs_Err)
    if not (os.path.isdir(savepath)):
        os.mkdir(savepath)
    
    print("The RMSE over all LOO tests was {:.3f}".format(RMSE))
    print("The maximum error was {:.3f}".format(maxErr))
    
    notelist=list()
    notelist.append("RMSE: %3.2f" % RMSE)
    kwargs=dict()
    kwargs['xlabel'] = xlabel
    kwargs['ylabel'] = ylabel
    kwargs['notelist'] = notelist
    kwargs['savepath'] = savepath
    kwargs['stepsize'] = stepsize

    plotpm.single(Ydata, Run_Y_Pred, **kwargs)
    
    if numericlabelfield == None:
        numericlabelfield = data.x_features[0]

    labels = np.asarray(data.get_data(numericlabelfield)).ravel()
    headerline = "%s,Measured,Predicted,Absolute error" % numericlabelfield 
    myarray = np.array([labels, Ydata, Run_Y_Pred, Run_Abs_Err]).transpose()
    
    csvname = os.path.join(savepath,"LOO_CV_data.csv")
    ptools.array_to_csv(csvname, headerline, myarray)
    return
