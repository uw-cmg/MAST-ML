import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error

def lowflux (model,data):
    data.remove_all_filters()
    data.add_exclusive_filter("log(flux)", '<', 11)
    trainX = np.asarray(data.get_x_data())
    trainY = np.asarray(data.get_y_data()).ravel()
    model.fit(trainX, trainY)

    data.remove_all_filters()
    data.add_inclusive_filter("log(flux)", '<', 11)
    data.add_exclusive_filter("log(fluence)", '<', 18.5)

    Ypredict = model.predict(np.asarray(data.get_x_data()))
    Yactual = np.asarray(data.get_y_data()).ravel()

    print(len(Yactual))
    return (Ypredict,Yactual)

def highfluence(model,data):
    data.remove_all_filters()
    data.add_exclusive_filter("log(fluence)", '>', 19.5)
    trainX = np.asarray(data.get_x_data())
    trainY = np.asarray(data.get_y_data()).ravel()
    model.fit(trainX, trainY)

    data.remove_all_filters()
    data.add_inclusive_filter("log(fluence)", '>', 19.5)
    data.add_exclusive_filter("log(flux)", '>', 13.5)

    Ypredict = model.predict(np.asarray(data.get_x_data()))
    Yactual = np.asarray(data.get_y_data()).ravel()

    print(len(Yactual))
    return (Ypredict, Yactual)

def execute(model,data,savepath,*args,**kwargs):

    lowfluxYpredict, lowfluxYactual = lowflux(model,data)
    highfluenceYpredict, highfluenceYactual = highfluence(model,data)

    lowfluxrms = np.sqrt(mean_squared_error(lowfluxYpredict, lowfluxYactual))
    highfluencerms = np.sqrt(mean_squared_error(highfluenceYpredict, highfluenceYactual))

    fig,ax = plt.subplots()
    ax.scatter(lowfluxYactual, lowfluxYpredict, s=7, lw=0)
    ax.plot(ax.get_ylim(), ax.get_ylim(), ls="--", c=".3")
    ax.text(.05, .83, 'RMS: {:.3f}'.format(lowfluxrms), fontsize=14, transform=ax.transAxes)
    ax.set_title("Low Flux IVAR Test")
    ax.set_ylabel("Predicted ∆sigma")
    ax.set_xlabel("Measured ∆sigma")
    fig.savefig(savepath.format(ax.get_title()), dpi=300, bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots()
    ax.scatter(highfluenceYactual, highfluenceYpredict, s=7, lw=0)
    ax.plot(ax.get_ylim(), ax.get_ylim(), ls="--", c=".3")
    ax.text(.05, .83, 'RMS: {:.3f}'.format(highfluencerms), fontsize=14, transform=ax.transAxes)
    ax.set_title("High Fluence IVAR Test")
    ax.set_ylabel("Predicted ∆sigma")
    ax.set_xlabel("Measured ∆sigma")
    fig.savefig(savepath.format(ax.get_title()), dpi=300, bbox_inches='tight')
    plt.close()