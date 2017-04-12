#!/usr/bin/env python
####
# test date and time
####

import time
import datetime
import matplotlib.dates
import calendar
import os

def test_time():
    timestr = "09/03/15 12:25"
    print("Input time: ", timestr)
    fmtstr = "%m/%d/%y %H:%M"
    timezone = time.timezone
    timezonename = time.tzname
    structtime = time.strptime(timestr, fmtstr)
    print("Time structure: ", structtime)
    print("Time zone: ", timezone, timezonename)
    epochsec = time.mktime(structtime)
#structtimeadj = time.gmtime(epochsec)
#epochsecadj = calendar.timegm(structtimeadj)
#print("Adjusted time struct: ", structtimeadj)
    epochsecadj = epochsec - timezone
    print("Epoch seconds: ", epochsec)
    print("Adjusted epoch seconds: ", epochsecadj)
    isdaylight = time.localtime().tm_isdst
    print("Is daylight savings?: ", isdaylight)
    if isdaylight:
        epochsecadj = epochsecadj + 3600.0
    print("Adjusted epoch seconds for daylight savings: ", epochsecadj)

    formatter = matplotlib.dates.DateFormatter(fmtstr)
#epochdays = epochsec / 3600.0/24.0
    matplotnum = matplotlib.dates.epoch2num(epochsecadj)
    print("Output date: ", matplotnum)
    dateout = formatter.format_data(matplotnum)
#dateout = formatter.format_data(epochdays)
    print("Output date: ", dateout)
    return

def test_plot():
    import plot_data.plot_xy as plotxy
    xdata=[1441301100,1441401100,1441501100,1441601100,1441701100]
    ydata=[1,2,3,4,5]
    plotxy.single(xdata,ydata,savepath=os.getcwd())
    plotxy.single(xdata,ydata,savepath=os.getcwd(),xlabel="X adj",timex="%m/%d/%y %H:%M")
    return

if __name__ == "__main__":
    test_time()
    test_plot()
