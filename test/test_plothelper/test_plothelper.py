#!/usr/bin/env python

from plot_data.PlotHelper import PlotHelper
import numpy as np
import os
xtest=np.arange(-10,10,1)
ytest=np.sin(xtest)
ytest2=np.cos(xtest)
xdatalist=[xtest,xtest]
ydatalist=[ytest,ytest2]
xerrlist=[None,None]
yerrlist=[np.ones(len(ytest))*0.01,np.ones(len(ytest2))*0.02]
kwargs=dict()
kwargs['xdatalist'] = xdatalist
kwargs['ydatalist'] = ydatalist
kwargs['xerrlist'] = xerrlist
kwargs['yerrlist'] = yerrlist
kwargs['labellist'] = ['sine','cosine']
kwargs['marklargest'] = "2,3"
kwargs['mlabellist'] = None
kwargs['guideline'] = 1
kwargs['xlabel'] = "Number"
kwargs['ylabel'] = "Function value"
kwargs['plotlabel'] = "sine_cos_overlay"
kwargs['save_path'] = os.path.join(os.getcwd(),"save_testing")
kwargs['notelist'] = ["Test plot"]
myph = PlotHelper(**kwargs)
#myph.multiple_overlay()
myph2 = PlotHelper()
myph2.test_all()


