from sklearn.kernel_ridge import KernelRidge
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
n_samples, n_features = 10, 5
rng = np.random.RandomState(0)
y = rng.randn(n_samples)
X = rng.randn(n_samples, n_features)

myrange=[0.001,0.01,0.1,1.0]
adict=dict()
gdict=dict()

for myalpha in myrange:
    adict[myalpha]=dict()
    for mygamma in myrange:
        if not (mygamma) in gdict.keys():
            gdict[mygamma]=dict()
        clf = KernelRidge(alpha=myalpha, gamma=mygamma,
                    coef0=1, degree=3, kernel='rbf',
                    kernel_params=None)
        clf.fit(X, y) 
        predict = clf.predict(X)
        adict[myalpha][mygamma]=predict
        gdict[mygamma][myalpha]=predict


colors=['r','g','b','k','m','c']
symbols=['x','o','d','s']
matplotlib.rcParams.update({'font.size':16})
smallfont = 0.85*matplotlib.rcParams['font.size']

def plot_comparisons(mdict, flabel="gamma", slabel="alpha"):
    aidx=0
    fkeys = list(mdict.keys())
    fkeys.sort()
    for firstkey in fkeys:
        fig = plt.figure()
        plt.hold(True)
        pidx = 0
        skeys = list(mdict[firstkey].keys())
        skeys.sort()
        for secondkey in skeys:
            predict = mdict[firstkey][secondkey]
            symbol = symbols[pidx%len(symbols)]
            color = colors[pidx%len(colors)]
            plt.plot(predict, y,
                linestyle = "None",
                marker=symbol, 
                markeredgecolor=color, markeredgewidth=2, markerfacecolor="None",
                label='%s=%3.3f, %s=%3.3f' % (flabel, firstkey, slabel, secondkey))
            pidx = pidx + 1
        lgd=plt.legend(numpoints=1, fancybox=True, 
                        fontsize=smallfont, loc='upper left')
        lgd.get_frame().set_alpha(0.5)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.savefig("%s_comparison_%i" % (slabel,aidx), bbox_inches="tight")
        plt.close()
        aidx = aidx + 1
    return

def plot_vs_x(mdict, xidx=0, flabel="alpha", slabel="gamma"):
    aidx=0
    xseries=X[:,xidx]
    fkeys = list(mdict.keys())
    fkeys.sort()
    for firstkey in fkeys:
        fig = plt.figure()
        plt.hold(True)
        pidx = 0
        plt.plot(xseries, y,
            linestyle = "None",
            marker='o',
            markeredgecolor='k', markeredgewidth=2, markerfacecolor="k",
            label='Actual')
        skeys = list(mdict[firstkey].keys())
        skeys.sort()
        for secondkey in skeys:
            predict = mdict[firstkey][secondkey]
            symbol = symbols[pidx%len(symbols)]
            color = colors[pidx%len(colors)]
            plt.plot(xseries,predict,
                linestyle = "None",
                marker=symbol, 
                markeredgecolor=color, markeredgewidth=2, markerfacecolor="None",
                label='%s=%3.3f, %s=%3.3f' % (flabel, firstkey, slabel, secondkey))
            pidx = pidx + 1
        lgd=plt.legend(numpoints=1, fancybox=True, 
                        fontsize=smallfont, loc='upper left')
        lgd.get_frame().set_alpha(0.5)
        plt.xlabel("X value, feature %i" % xidx)
        plt.ylabel("y value, actual or predicted")
        plt.savefig("%s_vs_x_%i" % (slabel,aidx), bbox_inches="tight")
        plt.close()
        aidx = aidx + 1
    return



plot_comparisons(adict, "alpha","gamma")
plot_comparisons(gdict, "gamma","alpha")
plot_vs_x(adict, 0, "alpha","gamma")
plot_vs_x(gdict, 0, "gamma","alpha")

